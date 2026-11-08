import json
import os
import sys
import tempfile
import wave
import logging
import subprocess
import copy
from pathlib import Path

import numpy as np
import sounddevice as sd
from PySide6.QtCore import QObject, QThread, Qt, Signal
from PySide6.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from conversation_state import ConversationState
from intent_detector import detect_intent, parse_correction
from llm_service import LLMService
from memory_manager import MemoryManager
from session_storage import SessionStorage


ROOT = Path(__file__).resolve().parent
LOG_PATH = ROOT / "whisperer.log"
logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    encoding="utf-8",
)
logging.getLogger().handlers[0].flush()


def load_json(filename: str) -> dict:
    with (ROOT / filename).open("r", encoding="utf-8") as file:
        return json.load(file)


def to_simplified(text: str) -> str:
    """Normalize ASR/LLM text so the UI and TTS always use Simplified Chinese."""
    try:
        from opencc import OpenCC

        return OpenCC("t2s").convert(text)
    except ImportError:
        # The prompt still requests simplified Chinese if the optional package
        # has not been installed yet.
        return text


class RecordingError(RuntimeError):
    pass


class Recorder:
    """A small microphone recorder controlled by start/stop calls."""

    def __init__(self, audio_config: dict):
        self.sample_rate = int(audio_config.get("sample_rate", 16000))
        self.channels = int(audio_config.get("channels", 1))
        self.blocksize = int(audio_config.get("blocksize", 1024))
        self.stream = None
        self.frames = []

    def start(self):
        self.frames = []

        def callback(indata, _frames, _time, status):
            if status:
                print(f"Audio status: {status}", file=sys.stderr)
            self.frames.append(indata.copy())

        try:
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                blocksize=self.blocksize,
                dtype="int16",
                callback=callback,
            )
            self.stream.start()
        except Exception as exc:
            self.stream = None
            raise RecordingError(f"无法打开麦克风：{exc}") from exc

    def stop_to_wav(self) -> Path:
        if self.stream is None:
            raise RecordingError("当前没有正在进行的录音")
        self.stream.stop()
        self.stream.close()
        self.stream = None
        if not self.frames:
            raise RecordingError("没有录到音频，请检查麦克风")

        audio = np.concatenate(self.frames, axis=0)
        path = Path(tempfile.mkstemp(prefix="whisperer_", suffix=".wav")[1])
        with wave.open(str(path), "wb") as wav_file:
            wav_file.setnchannels(self.channels)
            wav_file.setsampwidth(2)  # int16
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(audio.tobytes())
        return path


class PipelineWorker(QObject):
    finished = Signal(str, str, object)
    failed = Signal(str)
    progress = Signal(str)
    tts_finished = Signal()
    tts_failed = Signal(str)

    def __init__(self, wav_path: Path, config: dict, prompt: dict, state_dict: dict):
        super().__init__()
        self.wav_path = wav_path
        self.config = config
        self.prompt = prompt
        self.state_dict = state_dict

    def run(self):
        try:
            logging.info("Pipeline started: wav=%s", self.wav_path)
            self.progress.emit("正在加载语音识别模型……首次运行可能需要较长时间")
            text = self.transcribe()
            logging.info("Transcription returned: %r", text)
            if not text.strip():
                raise RuntimeError("没有识别到有效语音，请再试一次")
            self.progress.emit("语音识别完成，正在请求大模型……")
            intent = detect_intent(text)
            state = ConversationState.from_dict(copy.deepcopy(self.state_dict))
            memory = MemoryManager(state, self.config)
            correction = parse_correction(text) if intent == "correction" else None

            if intent == "clear_memory":
                memory.clear()
                answer = "好的，当前对话记忆已经清除。"
            elif intent == "new_session":
                memory.new_session()
                answer = "好的，我们开始一个新的话题。"
            elif intent == "resume_session":
                answer = "好的，我们继续之前的对话。"
            else:
                if correction:
                    memory.add_correction(correction["wrong"], correction["correct"])
                    user_for_model = (
                        "用户正在纠正上一轮语音识别，请以后续说法为准。"
                        f"上一轮可能识别为：{correction['wrong']}。"
                        f"用户明确说正确内容是：{correction['correct']}。"
                    )
                else:
                    user_for_model = text
                context = memory.context_messages()
                memory.add_user(text)
                answer = self.ask_llm_with_context(context, user_for_model)
                memory.add_assistant(answer)
                if memory.needs_summary():
                    self.progress.emit("正在整理较早的对话记忆……")
                    state.summary = self.summarize_state(state)
                    memory.trim_after_summary()
            logging.info("LLM response received: %d chars", len(answer))
            self.progress.emit("大模型回答完成，正在语音播报……")
            self.finished.emit(text, answer, {
                "intent": intent,
                "correction": correction,
                "state": state.to_dict(),
            })
            self.speak(answer)
            self.tts_finished.emit()
        except Exception as exc:
            logging.exception("Pipeline failed")
            self.failed.emit(str(exc))
        finally:
            self.wav_path.unlink(missing_ok=True)

    def transcribe(self) -> str:
        settings = self.config.get("speech_recognition", {})
        model = settings.get("model", "small")
        model_path = settings.get("model_path", "").strip()
        model_download_root = settings.get("model_download_root", "models")
        hf_endpoint = settings.get("hf_endpoint", "").strip()
        device = settings.get("device", "auto")
        compute_type = settings.get("compute_type", "auto")

        # CTranslate2 contains native code. Run it outside the Qt process so a
        # native crash becomes a readable subprocess error instead of a GUI exit.
        request = {
            "wav_path": str(self.wav_path),
            "model": model,
            "model_path": model_path,
            "model_download_root": model_download_root,
            "hf_endpoint": hf_endpoint,
            "language": settings.get("language", "zh"),
            "device": device,
            "compute_type": compute_type,
        }
        logging.info("Starting ASR subprocess: %s", request)
        result = subprocess.run(
            [sys.executable, "-u", str(ROOT / "asr_worker.py"), json.dumps(request)],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=900,
        )
        if result.stdout:
            logging.info("ASR stdout: %s", result.stdout.strip())
        if result.stderr:
            logging.info("ASR stderr: %s", result.stderr.strip())
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip() or "没有返回错误信息"
            raise RuntimeError(
                f"语音识别子进程异常退出（代码 {result.returncode}）。\n{detail}\n"
                "请查看 whisperer.log；如果是 CTranslate2 原生错误，可在 config.json 中将 device 改为 cpu。"
            )
        try:
            output = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"语音识别返回格式错误：{result.stdout[-1000:]}") from exc
        if output.get("error"):
            raise RuntimeError(output["error"])
        return to_simplified(output.get("text", "").strip())

    @staticmethod
    def cuda_available() -> bool:
        try:
            import ctranslate2

            return "cuda" in ctranslate2.get_supported_compute_types("cuda")
        except Exception:
            return False

    def ask_llm_with_context(self, context: list[dict[str, str]], text: str) -> str:
        service = LLMService(self.config, self.prompt)
        return to_simplified(service.answer(context, text))

    def summarize_state(self, state: ConversationState) -> str:
        service = LLMService(self.config, self.prompt)
        return to_simplified(service.summarize(state))

    def speak(self, text: str):
        settings = self.config.get("tts", {})
        if not settings.get("enabled", True):
            return
        # 将 Windows TTS 放在独立进程中，避免 SAPI/COM 底层异常关闭 Qt 主进程。
        command = [
            sys.executable,
            str(ROOT / "tts_worker.py"),
            text,
            str(int(settings.get("rate", 180))),
            str(float(settings.get("volume", 1.0))),
        ]
        try:
            result = subprocess.run(
                command,
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                timeout=180,
            )
            if result.returncode != 0:
                raise RuntimeError(result.stderr.strip() or "TTS 子进程返回失败")
        except Exception as exc:
            logging.exception("TTS failed")
            self.tts_failed.emit(f"文字已显示，但语音播报失败：{exc}")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.config = load_json("config.json")
        self.prompt = load_json("prompt.json")
        self.recorder = Recorder(self.config.get("audio", {}))
        self.storage = SessionStorage(ROOT / "sessions")
        self.state = self.storage.load_latest() or ConversationState()
        self.memory = MemoryManager(self.state, self.config)
        self.thread = None
        self.worker = None
        self.is_recording = False

        self.setWindowTitle("Whisperer · 语音知识助手")
        self.resize(680, 560)
        self.build_ui()

    def build_ui(self):
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setSpacing(12)

        self.status_label = QLabel("准备就绪：点击大按钮开始录音")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("font-size: 16px; color: #555;")
        layout.addWidget(self.status_label)

        self.record_button = QPushButton("开始录音\n（点击或按回车）")
        self.record_button.setMinimumHeight(120)
        self.record_button.setStyleSheet(
            "font-size: 24px; font-weight: bold; border-radius: 16px;"
        )
        self.record_button.clicked.connect(self.toggle_recording)
        layout.addWidget(self.record_button)

        self.transcript = QTextEdit()
        self.transcript.setReadOnly(True)
        self.transcript.setPlaceholderText("这里显示语音识别文字和助手回答")
        layout.addWidget(self.transcript)
        self.setCentralWidget(central)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and not event.isAutoRepeat():
            self.toggle_recording()
            event.accept()
            return
        super().keyPressEvent(event)

    def toggle_recording(self):
        if self.thread is not None and self.thread.isRunning():
            return
        if not self.is_recording:
            try:
                self.recorder.start()
                self.is_recording = True
                self.record_button.setText("停止录音\n（点击或按回车）")
                self.status_label.setText("正在录音……再次点击或按回车结束")
            except RecordingError as exc:
                self.show_error(str(exc))
        else:
            self.finish_recording()

    def finish_recording(self):
        try:
            wav_path = self.recorder.stop_to_wav()
        except RecordingError as exc:
            self.reset_record_button()
            self.show_error(str(exc))
            return

        self.is_recording = False
        self.record_button.setEnabled(False)
        self.record_button.setText("处理中……")
        self.status_label.setText("正在识别语音、请求大模型并准备播报……")
        self.thread = QThread(self)
        self.worker = PipelineWorker(wav_path, self.config, self.prompt, self.state.to_dict())
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_finished)
        self.worker.failed.connect(self.on_failed)
        self.worker.progress.connect(self.on_progress)
        self.worker.tts_finished.connect(self.on_tts_finished)
        self.worker.tts_failed.connect(self.on_tts_failed)
        self.worker.failed.connect(self.thread.quit)
        self.worker.tts_finished.connect(self.thread.quit)
        self.worker.tts_failed.connect(self.thread.quit)
        self.thread.finished.connect(self.cleanup_worker)
        self.thread.start()

    def on_finished(self, text: str, answer: str, metadata: dict):
        intent = metadata.get("intent", "question")
        if intent == "clear_memory":
            self.memory.clear()
        elif intent == "new_session":
            self.memory.new_session()
        elif intent not in ("resume_session",):
            state_dict = metadata.get("state")
            if state_dict:
                self.state = ConversationState.from_dict(state_dict)
                self.memory = MemoryManager(self.state, self.config)
        self.storage.save(self.state)
        self.transcript.append(f"你：{text}\n助手：{answer}\n")

    def on_progress(self, message: str):
        self.status_label.setText(message)

    def on_tts_finished(self):
        self.status_label.setText("回答完成：点击按钮或按回车继续")

    def on_tts_failed(self, message: str):
        self.status_label.setText(message)

    def on_failed(self, message: str):
        self.status_label.setText("处理失败：请检查配置或设备")
        self.show_error(message)

    def cleanup_worker(self):
        if self.worker:
            self.worker.deleteLater()
        if self.thread:
            self.thread.deleteLater()
        self.worker = None
        self.thread = None
        self.reset_record_button()

    def reset_record_button(self):
        self.is_recording = False
        self.record_button.setEnabled(True)
        self.record_button.setText("开始录音\n（点击或按回车）")

    def show_error(self, message: str):
        QMessageBox.critical(self, "Whisperer", message)

    def closeEvent(self, event):
        if self.is_recording:
            try:
                self.recorder.stop_to_wav()
            except Exception:
                pass
        if self.thread and self.thread.isRunning():
            self.thread.quit()
            self.thread.wait(3000)
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
