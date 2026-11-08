from openai import OpenAI


class LLMService:
    def __init__(self, config: dict, prompt: dict):
        self.config = config
        self.prompt = prompt
        self.settings = self._find_settings()

    def _find_settings(self) -> dict:
        if isinstance(self.config.get("llm"), dict):
            return self.config["llm"]
        reserved = {"speech_recognition", "audio", "tts", "memory"}
        for key, value in self.config.items():
            if key not in reserved and isinstance(value, dict) and "api_key" in value:
                return value
        return {}

    def complete(self, messages: list[dict[str, str]], max_tokens: int | None = None) -> str:
        api_key = self.settings.get("api_key", "").strip()
        if not api_key or api_key == "sk-your-api-key-here":
            raise RuntimeError("请先在 config.json 中填入真实的大模型 API Key")
        client = OpenAI(
            api_key=api_key,
            base_url=self.settings.get("base_url", "https://api.openai.com/v1"),
        )
        response = client.chat.completions.create(
            model=self.settings.get("model", "gpt-4o-mini"),
            temperature=float(self.settings.get("temperature", 0.2)),
            max_tokens=int(max_tokens or self.settings.get("max_tokens", 300)),
            messages=messages,
        )
        return (response.choices[0].message.content or "").strip()

    def answer(self, context: list[dict[str, str]], user_text: str) -> str:
        messages = [{"role": "system", "content": self.prompt["system_prompt"]}]
        messages.extend(context)
        messages.append({"role": "user", "content": user_text})
        return self.complete(messages)

    def summarize(self, state) -> str:
        history = "\n".join(
            f"{item['role']}: {item['content']}" for item in state.recent_messages
        )
        prompt = (
            "请用中文简体字总结下面的对话，供下一轮语音聊天继续使用。\n"
            "保留：当前主题、已确认事实、用户偏好、术语纠正、未解决问题。"
            "只输出结构清晰的简短摘要，不要添加对话中没有的信息。\n\n"
            f"已有摘要：{state.summary}\n"
            f"术语纠正：{state.corrections}\n"
            f"最近对话：\n{history}"
        )
        return self.complete(
            [{"role": "system", "content": "你是对话记忆摘要器，必须使用中文简体字。"}, {"role": "user", "content": prompt}],
            max_tokens=500,
        )
