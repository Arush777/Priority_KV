from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from .telegram_client import TelegramMessage

AGENT_TAG_RE = re.compile(r"\[agent:([a-z0-9_\-]+)\]", re.I)
CLAIM_RE = re.compile(r"CLAIM\s+(?P<id>T?\d+|S\d+)\b", re.I)
DONE_RE = re.compile(r"DONE\s+(?P<id>T?\d+|S\d+)\b", re.I)
BLOCKED_RE = re.compile(r"BLOCKED\s+(?P<id>T?\d+|S\d+)\b", re.I)
PROPOSE_SCOPE_RE = re.compile(r"PROPOSE_SCOPE\b", re.I)
ACK_SCOPE_RE = re.compile(r"ACK_SCOPE\b", re.I)
RESUME_RE = re.compile(r"RESUME_BRIDGE\b", re.I)
ASK_RE = re.compile(r"@agent:([a-z0-9_\-]+)\b", re.I)


class ControlAction(str, Enum):
    STOP = "stop"
    RESUME = "resume"
    ACK_SCOPE = "ack_scope"
    NONE = "none"


@dataclass
class ParsedMessage:
    msg: TelegramMessage
    author_agent: str | None
    mentions: list[str]
    claims: list[str]
    dones: list[str]
    blocked: list[str]
    propose_scope: bool
    ack_scope: bool
    control: ControlAction

    @property
    def is_peer_agent(self) -> bool:
        return self.author_agent is not None

    @property
    def is_human(self) -> bool:
        return not self.msg.is_from_bot and self.author_agent is None


def parse_message(msg: TelegramMessage) -> ParsedMessage:
    text = msg.text
    m = AGENT_TAG_RE.search(text)
    author = m.group(1).lower() if m else None
    mentions = [x.lower() for x in ASK_RE.findall(text)]
    claims = [x.upper() for x in CLAIM_RE.findall(text)]
    dones = [x.upper() for x in DONE_RE.findall(text)]
    blocked = [x.upper() for x in BLOCKED_RE.findall(text)]
    propose_scope = bool(PROPOSE_SCOPE_RE.search(text))
    ack_scope = bool(ACK_SCOPE_RE.search(text))

    control = ControlAction.NONE
    if RESUME_RE.search(text):
        control = ControlAction.RESUME
    if ack_scope:
        control = ControlAction.ACK_SCOPE

    return ParsedMessage(
        msg=msg,
        author_agent=author,
        mentions=mentions,
        claims=claims,
        dones=dones,
        blocked=blocked,
        propose_scope=propose_scope,
        ack_scope=ack_scope,
        control=control,
    )


def detect_stop(text: str, keywords: list[str]) -> str | None:
    upper = text.upper()
    for kw in keywords:
        if kw.upper() in upper:
            return kw
    return None


def format_transcript(parsed: list[ParsedMessage], my_id: str) -> str:
    lines: list[str] = []
    for p in parsed:
        who = p.author_agent or (
            "bot" if p.msg.is_from_bot else (p.msg.from_username or "human")
        )
        directed = ""
        if my_id in p.mentions:
            directed = " [DIRECTED_AT_YOU]"
        elif p.mentions:
            directed = f" [to:{','.join(p.mentions)}]"
        lines.append(
            f"- id={p.msg.message_id} from={who}{directed}: {p.msg.text.strip()}"
        )
    return "\n".join(lines) if lines else "(no new messages)"


def build_agent_prompt(
    *,
    agent_id: str,
    peer_hint: str,
    github_repo: str,
    default_branch: str,
    branch_ns: str,
    scope_path: str,
    collab_path: str,
    transcript: str,
    memory_blob: str,
    require_scope_ack: bool,
    max_commits: int,
    resumed: bool,
) -> str:
    resume_note = (
        "You are CONTINUING a resumed Cursor agent session — treat prior tool/chat "
        "context as yours when present."
        if resumed
        else "This may be a fresh agent session — lean on memory files + Telegram ring."
    )
    return f"""You are agent `{agent_id}` collaborating on `{github_repo}` (PriorityKV / agent KV-cache research).

Your peer agent is approximately: `{peer_hint}`. Humans may also write in Telegram.
{resume_note}

## Sticky memory (read every tick)
{memory_blob}

## Mandatory reading (in workspace)
- `{collab_path}` — collaboration protocol
- `{scope_path}` — current project scope
- `docs/PRIORITYKV_IMPLEMENTATION_PLAN.md` — canon plan
- `docs/decisions.md` — durable decisions (APPEND when you lock a choice)
- `docs/collab_memory.md` — shared rolling notes (update your tick note via bridge; you may refine Open asks)
- Existing code under this repo

## Telegram context
{transcript}

## Your job this tick
1. Use sticky memory + Telegram ring — do not pretend amnesia about CLAIMs/ASKs already recorded.
2. Prefer messages directed at `@agent:{agent_id}` or unanswered peer ASKs.
3. Discuss/decide with the peer; when a decision is final, append one line to `docs/decisions.md` under Decided (and move it out of Open).
4. Implement agreed work; CLAIM before large new ownership.
5. Scope edits only via PROPOSE_SCOPE; wait for ACK_SCOPE when required ({require_scope_ack}).
6. Push under `{branch_ns}` only. Never force-push. Never push secrets / `.env`.
7. Max ~{max_commits} commits this tick.
8. Cluster/GPU jobs only on THIS account.

## Safety
- STOP_BRIDGE / HALT_AGENTS pause the bridge externally.
- If blocked, post BLOCKED <id> with reason.

## Output contract
Do useful repo work with tools. Finish by writing `state/last_status_{agent_id}.txt`:

```
[agent:{agent_id}] TICK
SUMMARY: <1-3 sentences>
ACTIONS: <bullets>
CLAIM: <ids or none>
DONE: <ids or none>
BLOCKED: <ids or none>
ASK: @agent:<peer> <question or none>
PROPOSE_SCOPE: <yes/no — if yes, describe>
DECISIONS_WRITTEN: <yes/no — if yes, quote the line added to docs/decisions.md>
NEXT: <what peer should do>
```

Keep SUMMARY and ASK concrete so the peer can act next tick without humans.
"""


PROTOCOL_CHEATSHEET = """
Message grammar (Telegram group):
  [agent:ID] ...          agent attribution (auto-prefixed by bridge)
  @agent:ID ...           direct a request at a specific agent
  CLAIM T12 / CLAIM S1    claim a task / workstream
  DONE T12                mark complete
  BLOCKED T12 reason      unblock needed
  PROPOSE_SCOPE ...       propose scope change
  ACK_SCOPE               human/peer accepts pending scope proposal
  RESUME_BRIDGE           human resumes after STOP
  STOP_BRIDGE / HALT_AGENTS  human emergency stop
"""
