from __future__ import annotations

import json
import queue
import sys
import threading
from pathlib import Path

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except ModuleNotFoundError as exc:
    if exc.name != "_tkinter":
        raise
    print(
        "当前 Python 环境不包含 Tkinter 图形界面支持。请换用带 Tk 的 Python，"
        "或安装对应版本的 python-tk 后再运行。",
        file=sys.stderr,
    )
    raise SystemExit(1) from exc


class TranslatorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("MalayCN 马来语中文翻译")
        self.geometry("820x560")
        self.minsize(720, 500)

        self.translator = None
        self.worker_queue: queue.Queue[tuple[str, str]] = queue.Queue()

        self.checkpoint_var = tk.StringVar(value=self._default_checkpoint_dir())
        self.status_var = tk.StringVar(value="请选择训练输出目录并加载模型")

        self._build_ui()
        self._poll_worker_queue()

    def _default_checkpoint_dir(self) -> str:
        for path in ("artifacts/malaycn", "artifacts/malaycn-demo"):
            if Path(path).exists():
                return path
        return "artifacts/malaycn-demo"

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        top = ttk.Frame(self, padding=16)
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(1, weight=1)

        ttk.Label(top, text="模型目录").grid(row=0, column=0, sticky="w")
        checkpoint_entry = ttk.Entry(top, textvariable=self.checkpoint_var)
        checkpoint_entry.grid(row=0, column=1, sticky="ew", padx=(10, 8))

        ttk.Button(top, text="选择", command=self.choose_checkpoint_dir).grid(row=0, column=2)
        self.load_button = ttk.Button(top, text="加载模型", command=self.load_model)
        self.load_button.grid(row=0, column=3, padx=(8, 0))

        body = ttk.Frame(self, padding=(16, 0, 16, 8))
        body.grid(row=1, column=0, sticky="nsew")
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(1, weight=1)

        ttk.Label(body, text="马来语输入").grid(row=0, column=0, sticky="w", pady=(0, 6))
        ttk.Label(body, text="中文输出").grid(row=0, column=1, sticky="w", padx=(10, 0), pady=(0, 6))

        self.source_text = tk.Text(body, wrap="word", height=12, undo=True)
        self.source_text.grid(row=1, column=0, sticky="nsew")
        self.source_text.insert("1.0", "terima kasih")

        self.target_text = tk.Text(body, wrap="word", height=12, state="disabled")
        self.target_text.grid(row=1, column=1, sticky="nsew", padx=(10, 0))

        actions = ttk.Frame(self, padding=(16, 0, 16, 8))
        actions.grid(row=2, column=0, sticky="ew")
        actions.columnconfigure(0, weight=1)

        self.translate_button = ttk.Button(actions, text="翻译", command=self.translate_text)
        self.translate_button.grid(row=0, column=0, sticky="w")
        ttk.Button(actions, text="清空", command=self.clear_text).grid(row=0, column=1, padx=(8, 0))

        status = ttk.Label(self, textvariable=self.status_var, anchor="w", padding=(16, 0, 16, 12))
        status.grid(row=3, column=0, sticky="ew")

    def choose_checkpoint_dir(self) -> None:
        selected = filedialog.askdirectory(title="选择训练输出目录")
        if selected:
            self.checkpoint_var.set(selected)
            self.translator = None
            self.status_var.set("模型目录已更新，请重新加载模型")

    def load_model(self) -> None:
        checkpoint_dir = self.checkpoint_var.get().strip()
        if not checkpoint_dir:
            messagebox.showwarning("缺少模型目录", "请先选择训练输出目录。")
            return

        self._set_busy(True, "正在加载模型...")
        threading.Thread(target=self._load_model_worker, args=(checkpoint_dir,), daemon=True).start()

    def _load_model_worker(self, checkpoint_dir: str) -> None:
        try:
            self.translator = LoadedTranslator(Path(checkpoint_dir))
            self.worker_queue.put(("loaded", "模型加载完成，可以开始翻译"))
        except ModuleNotFoundError as exc:
            if exc.name == "tensorflow":
                self.worker_queue.put(("error", "当前环境未安装 TensorFlow，请先执行 pip install -r requirements.txt"))
            else:
                self.worker_queue.put(("error", f"缺少依赖: {exc.name}"))
        except FileNotFoundError as exc:
            self.worker_queue.put(("error", f"模型文件不存在: {exc.filename}"))
        except Exception as exc:
            self.worker_queue.put(("error", f"模型加载失败: {exc}"))

    def translate_text(self) -> None:
        text = self.source_text.get("1.0", "end").strip()
        if not text:
            messagebox.showwarning("缺少输入", "请输入需要翻译的马来语文本。")
            return

        if self.translator is None:
            self.load_model()
            self.after(300, lambda: self._translate_after_loading(text))
            return

        self._set_busy(True, "正在翻译...")
        threading.Thread(target=self._translate_worker, args=(text,), daemon=True).start()

    def _translate_after_loading(self, text: str) -> None:
        if self.translator is None:
            if self.load_button["state"] == "disabled":
                self.after(300, lambda: self._translate_after_loading(text))
            return
        self.translate_text()

    def _translate_worker(self, text: str) -> None:
        try:
            result = self.translator.translate(text)
            self.worker_queue.put(("translated", result))
        except Exception as exc:
            self.worker_queue.put(("error", f"翻译失败: {exc}"))

    def _poll_worker_queue(self) -> None:
        try:
            while True:
                event, payload = self.worker_queue.get_nowait()
                if event == "loaded":
                    self._set_busy(False, payload)
                elif event == "translated":
                    self._set_target_text(payload)
                    self._set_busy(False, "翻译完成")
                elif event == "error":
                    self._set_busy(False, payload)
                    messagebox.showerror("错误", payload)
        except queue.Empty:
            pass

        self.after(100, self._poll_worker_queue)

    def _set_busy(self, busy: bool, status: str) -> None:
        state = "disabled" if busy else "normal"
        self.load_button.configure(state=state)
        self.translate_button.configure(state=state)
        self.status_var.set(status)

    def _set_target_text(self, text: str) -> None:
        self.target_text.configure(state="normal")
        self.target_text.delete("1.0", "end")
        self.target_text.insert("1.0", text)
        self.target_text.configure(state="disabled")

    def clear_text(self) -> None:
        self.source_text.delete("1.0", "end")
        self._set_target_text("")
        self.status_var.set("已清空")


class LoadedTranslator:
    def __init__(self, checkpoint_dir: Path):
        import tensorflow as tf

        from .model import MalayChineseTransformer
        from .translate import build_vectorizer

        self.tf = tf
        self.checkpoint_dir = checkpoint_dir
        config_path = checkpoint_dir / "config.json"
        vocab_path = checkpoint_dir / "vocab.json"
        weights_path = checkpoint_dir / "model.weights.h5"

        config = json.loads(config_path.read_text(encoding="utf-8"))
        vocab = json.loads(vocab_path.read_text(encoding="utf-8"))

        self.max_target_len = config["max_target_len"]
        self.source_vectorizer = build_vectorizer(vocab["source"], config["max_source_len"])
        self.target_vectorizer = build_vectorizer(vocab["target"], config["target_vectorize_len"])
        self.target_index_lookup = dict(enumerate(vocab["target"]))

        self.model = MalayChineseTransformer(**config)
        dummy_source = tf.zeros((1, config["max_source_len"]), dtype=tf.int64)
        dummy_target = tf.zeros((1, config["max_target_len"]), dtype=tf.int64)
        self.model([dummy_source, dummy_target], training=False)
        self.model.load_weights(str(weights_path))

    def translate(self, text: str) -> str:
        from .translate import translate

        return translate(
            text,
            self.model,
            self.source_vectorizer,
            self.target_vectorizer,
            self.target_index_lookup,
            max_decoded_len=self.max_target_len,
        )


def main():
    app = TranslatorApp()
    app.mainloop()


if __name__ == "__main__":
    main()
