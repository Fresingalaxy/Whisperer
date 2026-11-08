from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any
import uuid


@dataclass
class ConversationState:
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    turn_index: int = 0
    stage: str = "idle"
    summary: str = ""
    topic: str = ""
    recent_messages: list[dict[str, str]] = field(default_factory=list)
    corrections: list[dict[str, str]] = field(default_factory=list)
    last_transcript: str = ""
    last_answer: str = ""

    def touch(self):
        self.updated_at = datetime.now().isoformat(timespec="seconds")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConversationState":
        allowed = {field_name for field_name in cls.__dataclass_fields__}
        return cls(**{key: value for key, value in data.items() if key in allowed})
