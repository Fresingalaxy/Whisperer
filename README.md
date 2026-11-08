# Whisperer 最小可行语音聊天助手

这是一个最小的 PySide6 语音聊天界面，流程为：

```text
点击大按钮 / 按回车开始录音
        ↓
点击大按钮 / 按回车停止录音
        ↓
faster-whisper 本地识别
        ↓
OpenAI 兼容大模型 API 格式化为简短回答
        ↓
pyttsx3 本地语音播报
```

## 安装

建议使用 Python 3.10 或更高版本：

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

首次运行 `faster-whisper` 时会自动下载配置中的模型。`small` 模型适合先验证流程；如果显卡和内存足够，可以在 `config.json` 中改为 `medium` 或 `large-v3-turbo`。

## 配置

### 大模型 API

编辑 `config.json`：

```json
{
  "llm": {
    "api_key": "填入你的真实 API Key",
    "base_url": "https://api.openai.com/v1",
    "model": "gpt-4o-mini"
  }
}
```

`base_url` 和 `model` 支持大多数 OpenAI 兼容服务。当前文件中的 `sk-your-api-key-here` 只是占位符，程序会在使用前提示修改。

程序也兼容把大模型配置写在自定义名称下，例如 `deepseek-v4.1-flash`；只要该配置段包含 `api_key` 和 `base_url` 即可。建议后续统一改成 `llm` 配置段，结构更直观。

### 提示词

编辑 `prompt.json` 中的 `system_prompt`，即可调整回答长度、格式和语气，不需要修改 Python 代码。

### 语音识别

在 `config.json` 的 `speech_recognition` 中调整：

- `model`: `small`、`medium` 或 `large-v3-turbo`
- `device`: `cpu` 或 `cuda`，`auto` 会自动选择
- `compute_type`: CPU 通常使用 `int8`，GPU 可以使用 `float16`
- `model_path`: 如果已经手动下载 Whisper 模型，可以填写本地模型目录，程序将不再联网下载
- `model_download_root`: 模型缓存目录，默认为项目下的 `models`
- `hf_endpoint`: Hugging Face 镜像地址；网络可访问官方站点时可以留空

如果看到 `ConnectTimeout [WinError 10060]` 和 `cannot find the appropriate snapshots folder`，说明失败发生在 Whisper 模型下载阶段，还没有执行大模型 API 请求。可先保留默认的 `https://hf-mirror.com` 重试；如果镜像仍不可访问，请手动下载对应模型并填写 `model_path`。

程序运行日志会写入项目目录下的 `whisperer.log`。界面会先显示识别文字和大模型文字回答，再进行语音播报；TTS 在独立子进程中运行，避免 Windows 语音引擎异常导致 Qt 界面退出。

语音识别也在 `asr_worker.py` 独立子进程中运行。这样如果 CTranslate2、CUDA 或音频相关原生库异常退出，主界面不会无提示关闭，错误代码和子进程输出会写入 `whisperer.log`。当前 `device: auto` 在 ASR 子进程中优先使用 CPU，以提高 MVP 稳定性；确认流程稳定后再切换到 CUDA。

## 运行

```powershell
python app.py
```

点击大按钮第一次开始录音，第二次停止录音；回车键具有相同效果。处理期间按钮会被禁用，语音播报完成后可以开始下一轮。

## Windows 语音播报说明

`pyttsx3` 使用 Windows 本地语音引擎。若中文播报不自然，可以在 Windows 的“语音”设置中安装中文语音包，再在系统语音设置里选择合适的声音。

程序会通过 `opencc-python-reimplemented` 将语音识别文字和大模型回答统一转换为中文简体字；同时 `prompt.json` 也明确要求大模型使用简体中文。

## 语音上下文记忆

程序会自动保存会话到 `sessions/`，默认启动时恢复最近一次会话。每次请求会携带会话摘要、最近 8 轮对话和已确认的语音纠正。对话较长时会自动压缩较早内容为摘要。

可以直接用语音控制记忆：

- “我刚才说的是强化学习，不是强化力量”：记录语音纠正，后续以正确说法为准
- “清除当前记忆”：清空当前会话记忆
- “开始新话题”：创建新的会话上下文
- “继续上次对话”：继续使用已恢复的会话

这些操作都不要求手动输入文字；界面文本仅用于辅助查看状态。
