from conversation_state import ConversationState


class MemoryManager:
    def __init__(self, state: ConversationState, config: dict):
        self.state = state
        memory = config.get("memory", {})
        self.window_turns = int(memory.get("window_turns", 8))
        self.summary_trigger_chars = int(memory.get("summary_trigger_chars", 6000))
        self.keep_after_summary = int(memory.get("keep_after_summary", 4))

    def add_user(self, text: str):
        self.state.recent_messages.append({"role": "user", "content": text})
        self.state.last_transcript = text
        self.state.turn_index += 1
        self.state.touch()

    def add_assistant(self, text: str):
        self.state.recent_messages.append({"role": "assistant", "content": text})
        self.state.last_answer = text
        self.state.touch()

    def add_correction(self, wrong: str, correct: str):
        self.state.corrections.append({"wrong": wrong, "correct": correct})
        self.state.touch()

    def clear(self):
        self.state.summary = ""
        self.state.topic = ""
        self.state.recent_messages.clear()
        self.state.corrections.clear()
        self.state.last_transcript = ""
        self.state.last_answer = ""
        self.state.turn_index = 0
        self.state.touch()

    def new_session(self):
        self.clear()
        self.state.session_id = __import__("uuid").uuid4().hex

    def needs_summary(self) -> bool:
        message_chars = sum(len(item["content"]) for item in self.state.recent_messages)
        return len(self.state.recent_messages) > self.window_turns * 2 or message_chars > self.summary_trigger_chars

    def context_messages(self) -> list[dict[str, str]]:
        messages = []
        if self.state.summary:
            messages.append({"role": "system", "content": f"会话历史摘要：\n{self.state.summary}"})
        if self.state.corrections:
            corrections = "；".join(
                f"{item['wrong']} → {item['correct']}" for item in self.state.corrections[-10:]
            )
            messages.append({"role": "system", "content": f"已确认的语音纠正：{corrections}"})
        messages.extend(self.state.recent_messages[-self.window_turns * 2 :])
        return messages

    def trim_after_summary(self):
        self.state.recent_messages = self.state.recent_messages[-self.keep_after_summary * 2 :]
