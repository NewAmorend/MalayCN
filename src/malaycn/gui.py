from __future__ import annotations

import queue
import sys
import threading

try:
    import tkinter as tk
    from tkinter import messagebox, ttk
except ModuleNotFoundError as exc:
    if exc.name != "_tkinter":
        raise
    print(
        "当前 Python 环境不包含 Tkinter 图形界面支持。请换用带 Tk 的 Python，"
        "或安装对应版本的 python-tk 后再运行。",
        file=sys.stderr,
    )
    raise SystemExit(1) from exc

from .pretrained import DEFAULT_MODEL_NAME, M2M100Translator


class TranslatorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("MalayCN 马来语中文翻译")
        self.geometry("820x560")
        self.minsize(720, 500)

        self.translator: M2M100Translator | None = None
        self.worker_queue: queue.Queue[tuple[str, str]] = queue.Queue()

        self.model_name_var = tk.StringVar(value=DEFAULT_MODEL_NAME)
        self.source_lang_var = tk.StringVar(value="ms")
        self.target_lang_var = tk.StringVar(value="zh")
        self.status_var = tk.StringVar(value="点击加载模型。首次运行会从 Hugging Face 下载模型。")

        self._build_ui()
        self._poll_worker_queue()

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        top = ttk.Frame(self, padding=16)
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(1, weight=1)

        ttk.Label(top, text="模型").grid(row=0, column=0, sticky="w")
        model_entry = ttk.Entry(top, textvariable=self.model_name_var)
        model_entry.grid(row=0, column=1, sticky="ew", padx=(10, 8))

        lang_frame = ttk.Frame(top)
        lang_frame.grid(row=0, column=2, sticky="e", padx=(0, 8))
        ttk.Label(lang_frame, text="源").grid(row=0, column=0)
        ttk.Entry(lang_frame, width=5, textvariable=self.source_lang_var).grid(row=0, column=1, padx=(4, 8))
        ttk.Label(lang_frame, text="目标").grid(row=0, column=2)
        ttk.Entry(lang_frame, width=5, textvariable=self.target_lang_var).grid(row=0, column=3, padx=(4, 0))

        self.load_button = ttk.Button(top, text="加载模型", command=self.load_model)
        self.load_button.grid(row=0, column=3)

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

    def load_model(self) -> None:
        model_name = self.model_name_var.get().strip()
        source_lang = self.source_lang_var.get().strip()
        target_lang = self.target_lang_var.get().strip()

        if not model_name or not source_lang or not target_lang:
            messagebox.showwarning("缺少配置", "请填写模型名、源语言代码和目标语言代码。")
            return

        self._set_busy(True, "正在加载模型，首次运行可能需要下载几 GB 权重...")
        threading.Thread(
            target=self._load_model_worker,
            args=(model_name, source_lang, target_lang),
            daemon=True,
        ).start()

    def _load_model_worker(self, model_name: str, source_lang: str, target_lang: str) -> None:
        try:
            self.translator = M2M100Translator(
                model_name=model_name,
                source_lang=source_lang,
                target_lang=target_lang,
            )
            self.worker_queue.put(("loaded", "模型加载完成，可以开始翻译"))
        except ModuleNotFoundError as exc:
            self.worker_queue.put(("error", f"缺少依赖: {exc.name}。请先执行 pip install -r requirements.txt"))
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
        self._set_busy(True, "正在翻译...")
        threading.Thread(target=self._translate_worker, args=(text,), daemon=True).start()

    def _translate_worker(self, text: str) -> None:
        try:
            assert self.translator is not None
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


def main():
    app = TranslatorApp()
    app.mainloop()


if __name__ == "__main__":
    main()

