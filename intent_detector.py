import re


CORRECTION_PATTERNS = (
    r"我(?:刚才|前面)?说的是(?P<correct>.+?)(?:，|,|。|不是|而不是)(?P<wrong>.+)?$",
    r"不是(?P<wrong>.+?)[，,。]?(?:是|应该是)(?P<correct>.+)$",
    r"(?:纠正一下|更正一下)[，,：:]?(?P<correct>.+)$",
    r"刚才(?:识别|听)错了[，,：:]?(?P<correct>.+)$",
    r"(?P<wrong>.+?)[，,]不是这个词[，,]?应该是(?P<correct>.+)$",
)


def detect_intent(text: str) -> str:
    normalized = text.strip()
    if any(marker in normalized for marker in ("清除当前记忆", "忘掉刚才", "忘记这段对话")):
        return "clear_memory"
    if any(marker in normalized for marker in ("开始新话题", "开启新话题", "重新开始对话")):
        return "new_session"
    if any(re.search(pattern, normalized) for pattern in CORRECTION_PATTERNS):
        return "correction"
    if any(marker in normalized for marker in ("继续上次对话", "恢复上次对话")):
        return "resume_session"
    return "question"


def parse_correction(text: str) -> dict[str, str]:
    normalized = text.strip()
    for pattern in CORRECTION_PATTERNS:
        match = re.search(pattern, normalized)
        if match:
            groups = match.groupdict()
            wrong = (groups.get("wrong") or "上一轮识别内容").strip(" ，,。")
            wrong = re.sub(r"^(?:不是|而不是)\s*", "", wrong).strip(" ，,。")
            return {
                "wrong": wrong or "上一轮识别内容",
                "correct": (groups.get("correct") or normalized).strip(" ，,。"),
            }
    return {"wrong": "上一轮识别内容", "correct": normalized}
