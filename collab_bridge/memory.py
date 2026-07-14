from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .protocol import ParsedMessage, format_transcript
from .telegram_client import TelegramMessage, message_to_dict


class CollabMemory:
    """Sticky context across ticks: Telegram ring + rolling summary + Cursor agent id."""

    def __init__(self, state_dir: Path, repo_root: Path, agent_id: str, window: int = 50):
        self.state_dir = state_dir
        self.repo_root = repo_root
        self.agent_id = agent_id
        self.window = max(5, window)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        (self.repo_root / "docs").mkdir(parents=True, exist_ok=True)

        self.ring_path = self.state_dir / f"telegram_ring_{agent_id}.json"
        self.summary_path = self.state_dir / f"collab_memory_{agent_id}.md"
        self.shared_memory_path = self.repo_root / "docs" / "collab_memory.md"
        self.decisions_path = self.repo_root / "docs" / "decisions.md"
        # Separate from bridge_*.json so offset/pause state is never clobbered
        self.cursor_id_path = self.state_dir / f"cursor_agent_{agent_id}.txt"

        self._ensure_docs()

    def _ensure_docs(self) -> None:
        if not self.decisions_path.exists():
            self.decisions_path.write_text(
                "# Decisions log (append-only)\n\n"
                "Agents and humans append one line per durable decision.\n"
                "Format: `YYYY-MM-DD | who | decision`\n\n"
                "## Open\n\n"
                "- (none yet)\n\n"
                "## Decided\n\n",
                encoding="utf-8",
            )
        if not self.shared_memory_path.exists():
            self.shared_memory_path.write_text(
                "# Shared collab memory (Priority_KV)\n\n"
                "Updated by each agent tick. Read this every tick.\n"
                "Per-agent detail also lives in `state/collab_memory_<id>.md` (local).\n\n"
                "## Current picture\n\n"
                "- Project: Priority_KV / PriorityKV-Agent\n"
                "- Bridges: arush + friend online\n"
                "- Memory upgrade: sticky ring + resume + decisions.md\n\n"
                "## Open asks\n\n"
                "- Friend asked arush: PriorityBench data layout — committed JSONL under "
                "`data/prioritybench/{calibration,validation,test}/` + generator, "
                "or generator+seeds only with JSONL gitignored?\n"
                "- Arush asked friend: start tool-schema category first? Which Qwen3-8B "
                "chat-template version to pin for page tagging?\n\n"
                "## Recent tick notes\n\n",
                encoding="utf-8",
            )
        if not self.summary_path.exists():
            self.summary_path.write_text(
                f"# Local collab memory — agent:{self.agent_id}\n\n"
                "(filled after ticks)\n",
                encoding="utf-8",
            )

    def _load_ring(self) -> list[dict[str, Any]]:
        if not self.ring_path.exists():
            return []
        try:
            data = json.loads(self.ring_path.read_text(encoding="utf-8"))
            return list(data.get("messages") or [])
        except Exception:
            return []

    def _save_ring(self, messages: list[dict[str, Any]]) -> None:
        payload = {
            "updated_at": time.time(),
            "window": self.window,
            "messages": messages[-self.window :],
        }
        tmp = self.ring_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(self.ring_path)

    def append_telegram_messages(self, messages: list[TelegramMessage]) -> list[dict[str, Any]]:
        ring = self._load_ring()
        seen = {(m.get("update_id"), m.get("message_id")) for m in ring}
        for msg in messages:
            item = message_to_dict(msg)
            key = (item.get("update_id"), item.get("message_id"))
            if key in seen:
                continue
            ring.append(item)
            seen.add(key)
        ring = ring[-self.window :]
        self._save_ring(ring)
        return ring

    def ring_as_transcript(self, my_id: str) -> str:
        """Rebuild ParsedMessage-like lines from stored ring dicts."""
        ring = self._load_ring()
        if not ring:
            return "(telegram ring empty)"
        lines: list[str] = []
        for item in ring:
            text = (item.get("text") or "").strip()
            if not text:
                continue
            # light parse for author tag
            author = None
            if text.startswith("[agent:"):
                try:
                    author = text.split("]", 1)[0].split(":", 1)[1].lower()
                except Exception:
                    author = None
            who = author or (
                "bot"
                if item.get("from_is_bot")
                else (item.get("from_username") or "human")
            )
            mid = item.get("message_id")
            lines.append(f"- id={mid} from={who}: {text}")
        return "\n".join(lines) if lines else "(telegram ring empty)"

    def read_summary(self) -> str:
        local = (
            self.summary_path.read_text(encoding="utf-8")
            if self.summary_path.exists()
            else ""
        )
        shared = (
            self.shared_memory_path.read_text(encoding="utf-8")
            if self.shared_memory_path.exists()
            else ""
        )
        decisions = (
            self.decisions_path.read_text(encoding="utf-8")
            if self.decisions_path.exists()
            else ""
        )
        return (
            "### Shared docs/collab_memory.md\n"
            f"{shared[-4000:]}\n\n"
            "### Local rolling summary\n"
            f"{local[-2500:]}\n\n"
            "### docs/decisions.md\n"
            f"{decisions[-3000:]}\n"
        )

    def update_after_tick(self, status_body: str, *, new_msgs: int, resumed: bool) -> None:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        summary_line = ""
        for line in status_body.splitlines():
            if line.startswith("SUMMARY:"):
                summary_line = line[len("SUMMARY:") :].strip()
                break
        block = (
            f"\n### {ts} agent:{self.agent_id} "
            f"(new_msgs={new_msgs}, resumed={resumed})\n"
            f"{summary_line or status_body[:500]}\n"
        )
        prev = self.summary_path.read_text(encoding="utf-8") if self.summary_path.exists() else ""
        self.summary_path.write_text((prev + block)[-12000:], encoding="utf-8")

        # Append a short note under shared memory Recent tick notes
        shared = (
            self.shared_memory_path.read_text(encoding="utf-8")
            if self.shared_memory_path.exists()
            else "# Shared collab memory\n\n## Recent tick notes\n\n"
        )
        note = f"- {ts} `{self.agent_id}`: {summary_line or '(see Telegram TICK)'}\n"
        if "## Recent tick notes" in shared:
            shared = shared.replace(
                "## Recent tick notes\n\n",
                "## Recent tick notes\n\n" + note,
                1,
            )
        else:
            shared = shared.rstrip() + "\n\n## Recent tick notes\n\n" + note
        # keep file bounded
        if len(shared) > 20000:
            shared = shared[:2000] + "\n\n...\n\n" + shared[-16000:]
        self.shared_memory_path.write_text(shared, encoding="utf-8")

    # --- Cursor agent id persistence (for Agent.resume) ---

    def get_cursor_agent_id(self) -> str | None:
        if not self.cursor_id_path.exists():
            return None
        val = self.cursor_id_path.read_text(encoding="utf-8").strip()
        return val or None

    def set_cursor_agent_id(self, agent_id: str | None) -> None:
        if not agent_id:
            if self.cursor_id_path.exists():
                self.cursor_id_path.unlink()
            return
        self.cursor_id_path.write_text(agent_id.strip() + "\n", encoding="utf-8")


def format_new_and_ring(
    new_parsed: list[ParsedMessage],
    ring_transcript: str,
    my_id: str,
) -> str:
    new_part = format_transcript(new_parsed, my_id) if new_parsed else "(no brand-new messages)"
    return (
        "## Brand-new since last poll\n"
        f"{new_part}\n\n"
        f"## Last N Telegram messages (sticky ring)\n"
        f"{ring_transcript}\n"
    )
