"""Paged mixed-precision KV manager (W2): allocate, tag, demote under budget.

CPU reference controller — no CUDA. Real kernels attach later (W3+).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from prioritykv.byte_model import (
    DEFAULT_ALLOC_UNIT_TOKENS,
    PHYSICAL_PAGE_TOKENS,
    QWEN3_8B_KV,
    ModelKvGeom,
    realized_bytes,
)
from prioritykv.page_roles import (
    HARD_PROTECTED_ROLES,
    PROTECTED_ROLES,
    PageRole,
    StorageDtype,
)
from prioritykv.tagging import tag_chat_to_token_roles


@dataclass
class Page:
    page_id: int
    start_token: int
    n_tokens: int  # 1..PHYSICAL_PAGE_TOKENS
    role: PageRole
    dtype: StorageDtype = StorageDtype.BF16

    @property
    def end_token(self) -> int:
        return self.start_token + self.n_tokens


@dataclass
class PageManagerConfig:
    page_tokens: int = PHYSICAL_PAGE_TOKENS
    alloc_unit_tokens: int = DEFAULT_ALLOC_UNIT_TOKENS
    recent_window: int = 128
    sink_tokens: int = 16
    budget_frac: float = 0.50
    geom: ModelKvGeom = QWEN3_8B_KV
    # How often generated pages are reconsidered for demotion (plan §4.1).
    demote_every_tokens: int = 128


@dataclass
class PageManager:
    """Owns the page table for one sequence."""

    config: PageManagerConfig = field(default_factory=PageManagerConfig)
    pages: List[Page] = field(default_factory=list)
    _next_id: int = 0
    _tokens_since_demote: int = 0

    # --- queries -----------------------------------------------------------

    @property
    def seq_len(self) -> int:
        if not self.pages:
            return 0
        return self.pages[-1].end_token

    def dtype_token_counts(self) -> Dict[StorageDtype, int]:
        counts = {StorageDtype.BF16: 0, StorageDtype.INT4: 0}
        for p in self.pages:
            counts[p.dtype] += p.n_tokens
        return counts

    def realized_bytes(self) -> int:
        c = self.dtype_token_counts()
        return realized_bytes(
            num_bf16_tokens=c[StorageDtype.BF16],
            num_int4_tokens=c[StorageDtype.INT4],
            num_kv_heads=self.config.geom.num_kv_heads,
            head_dim=self.config.geom.head_dim,
            page_tokens=self.config.page_tokens,
            num_layers=self.config.geom.num_layers,
        )

    def budget_bytes(self) -> int:
        full = realized_bytes(
            num_bf16_tokens=max(self.seq_len, 1),
            num_int4_tokens=0,
            num_kv_heads=self.config.geom.num_kv_heads,
            head_dim=self.config.geom.head_dim,
            page_tokens=self.config.page_tokens,
            num_layers=self.config.geom.num_layers,
        )
        # Budget relative to FullKV at current length.
        return int(full * self.config.budget_frac)

    def within_budget(self) -> bool:
        if self.seq_len == 0:
            return True
        return self.realized_bytes() <= self.budget_bytes()

    # --- construction from chat -------------------------------------------

    def build_from_messages(self, messages: Sequence[dict]) -> None:
        """Replace page table from a chat trace (approx token roles)."""
        roles = tag_chat_to_token_roles(
            messages,
            recent_window=self.config.recent_window,
            sink_tokens=self.config.sink_tokens,
        )
        self.pages.clear()
        self._next_id = 0
        self._tokens_since_demote = 0
        if not roles:
            return

        # Pack into physical pages; page role = majority vote (ties → more protected).
        pt = self.config.page_tokens
        for start in range(0, len(roles), pt):
            chunk = roles[start : start + pt]
            role = _majority_role(chunk)
            dtype = (
                StorageDtype.BF16
                if role in PROTECTED_ROLES
                else StorageDtype.INT4
            )
            self.pages.append(
                Page(
                    page_id=self._next_id,
                    start_token=start,
                    n_tokens=len(chunk),
                    role=role,
                    dtype=dtype,
                )
            )
            self._next_id += 1

        self.enforce_budget()

    # --- append / demotion -------------------------------------------------

    def append_generated_tokens(self, n: int, role: PageRole = PageRole.GENERATED) -> None:
        """Append ``n`` newly generated tokens (usually GENERATED)."""
        if n <= 0:
            return
        pt = self.config.page_tokens
        remaining = n
        while remaining > 0:
            take = min(pt, remaining)
            # If last page is partial and same role/dtype, extend it.
            if (
                self.pages
                and self.pages[-1].n_tokens < pt
                and self.pages[-1].role == role
                and self.pages[-1].dtype == StorageDtype.BF16
            ):
                room = pt - self.pages[-1].n_tokens
                add = min(room, remaining)
                self.pages[-1].n_tokens += add
                remaining -= add
                continue
            start = self.seq_len
            self.pages.append(
                Page(
                    page_id=self._next_id,
                    start_token=start,
                    n_tokens=take,
                    role=role,
                    dtype=StorageDtype.BF16,  # new tokens start hot
                )
            )
            self._next_id += 1
            remaining -= take

        self._retouch_recent_window()
        self._tokens_since_demote += n
        if self._tokens_since_demote >= self.config.demote_every_tokens:
            self._tokens_since_demote = 0
            self.demote_aged_generated()
        self.enforce_budget()

    def demote_aged_generated(self) -> int:
        """Demote generated pages outside the recent window to INT4. Returns count."""
        if not self.pages:
            return 0
        cutoff = max(0, self.seq_len - self.config.recent_window)
        n = 0
        for p in self.pages:
            if p.role == PageRole.GENERATED and p.end_token <= cutoff:
                if p.dtype != StorageDtype.INT4 and p.role not in HARD_PROTECTED_ROLES:
                    p.dtype = StorageDtype.INT4
                    n += 1
        return n

    def enforce_budget(self) -> int:
        """Demote eligible pages until within budget. Returns number demoted.

        Order: FILLER → GENERATED → OTHER → protected soft roles (never SINK).
        Linear risk score (W4) will replace this tie-break later.
        """
        demoted = 0
        order = (
            PageRole.FILLER,
            PageRole.GENERATED,
            PageRole.OTHER,
            PageRole.CONSTRAINT,
            PageRole.TOOL,
            PageRole.SYSTEM,
            PageRole.RECENT,
        )
        while not self.within_budget():
            victim: Optional[Page] = None
            for role in order:
                if role in HARD_PROTECTED_ROLES:
                    continue
                # Prefer oldest page of this role still in BF16.
                cands = [
                    p
                    for p in self.pages
                    if p.role == role and p.dtype == StorageDtype.BF16
                ]
                if cands:
                    victim = cands[0]
                    break
            if victim is None:
                break  # cannot demote further without violating hard protect
            victim.dtype = StorageDtype.INT4
            demoted += 1
        return demoted

    def check_invariants(self) -> List[str]:
        """Return list of invariant violations (empty = ok)."""
        errs: List[str] = []
        # Contiguity / page size.
        expect = 0
        for p in self.pages:
            if p.start_token != expect:
                errs.append(f"page {p.page_id} start {p.start_token} != {expect}")
            if not (1 <= p.n_tokens <= self.config.page_tokens):
                errs.append(f"page {p.page_id} bad n_tokens={p.n_tokens}")
            expect = p.end_token
            if p.role in HARD_PROTECTED_ROLES and p.dtype != StorageDtype.BF16:
                errs.append(f"sink page {p.page_id} demoted to {p.dtype}")
        if self.seq_len > 0 and not self.within_budget():
            # Allow tiny float slack? budgets are int — must hold.
            errs.append(
                f"over budget: realized={self.realized_bytes()} "
                f"budget={self.budget_bytes()}"
            )
        # Recent window pages should be BF16 when budget allows — soft check:
        # only flag if a RECENT page is INT4 while some FILLER is still BF16.
        filler_hot = any(
            p.role == PageRole.FILLER and p.dtype == StorageDtype.BF16 for p in self.pages
        )
        recent_cold = any(
            p.role == PageRole.RECENT and p.dtype == StorageDtype.INT4 for p in self.pages
        )
        if filler_hot and recent_cold:
            errs.append("RECENT demoted while FILLER still BF16")
        return errs

    def _retouch_recent_window(self) -> None:
        cutoff = max(0, self.seq_len - self.config.recent_window)
        for p in self.pages:
            if p.start_token >= cutoff and p.role in (
                PageRole.GENERATED,
                PageRole.FILLER,
                PageRole.OTHER,
            ):
                p.role = PageRole.RECENT
                if p.dtype == StorageDtype.INT4:
                    p.dtype = StorageDtype.BF16


def _majority_role(chunk: Sequence[PageRole]) -> PageRole:
    counts: Dict[PageRole, int] = {}
    for r in chunk:
        counts[r] = counts.get(r, 0) + 1
    # Prefer protected roles on ties.
    def key(item: tuple[PageRole, int]) -> tuple:
        role, c = item
        return (c, 1 if role in PROTECTED_ROLES else 0, 1 if role in HARD_PROTECTED_ROLES else 0)

    return max(counts.items(), key=key)[0]
