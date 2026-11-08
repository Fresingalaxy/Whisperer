import json
from pathlib import Path

from conversation_state import ConversationState


class SessionStorage:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, state: ConversationState):
        state.touch()
        path = self.root / f"{state.session_id}.json"
        path.write_text(json.dumps(state.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    def load_latest(self) -> ConversationState | None:
        files = sorted(self.root.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True)
        if not files:
            return None
        try:
            return ConversationState.from_dict(json.loads(files[0].read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return None
