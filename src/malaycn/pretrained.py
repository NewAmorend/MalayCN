from __future__ import annotations


DEFAULT_MODEL_NAME = "facebook/m2m100_418M"
DEFAULT_SOURCE_LANG = "ms"
DEFAULT_TARGET_LANG = "zh"


class M2M100Translator:
    """基于 Hugging Face M2M100 的马来语到中文翻译器。"""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        source_lang: str = DEFAULT_SOURCE_LANG,
        target_lang: str = DEFAULT_TARGET_LANG,
        device: str | None = None,
        local_files_only: bool = False,
    ):
        import torch
        from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

        self.torch = torch
        self.model_name = model_name
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = M2M100Tokenizer.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        )
        self.model = M2M100ForConditionalGeneration.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        )
        self.model.to(self.device)
        self.model.eval()

    def translate(self, text: str, max_length: int = 128, num_beams: int = 4) -> str:
        text = text.strip()
        if not text:
            return ""

        self.tokenizer.src_lang = self.source_lang
        encoded = self.tokenizer(text, return_tensors="pt", padding=True).to(self.device)

        with self.torch.no_grad():
            generated_tokens = self.model.generate(
                **encoded,
                forced_bos_token_id=self.tokenizer.get_lang_id(self.target_lang),
                max_length=max_length,
                num_beams=num_beams,
            )

        return self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

