import re
import unicodedata


_CJK_RE = re.compile(r"([\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff])")
_PUNCT_RE = re.compile(r"([,.!?;:，。！？；：、（）()\"'])")
_SPACE_RE = re.compile(r"\s+")


def normalize_text(text: str, lang: str) -> str:
    """把原始句子规整成空格分隔 token，便于 TextVectorization 处理。"""
    text = unicodedata.normalize("NFKC", str(text)).strip()
    text = _SPACE_RE.sub(" ", text)

    if lang == "ms":
        text = text.lower()
        text = _PUNCT_RE.sub(r" \1 ", text)
    elif lang == "zh":
        text = _CJK_RE.sub(r" \1 ", text)
        text = _PUNCT_RE.sub(r" \1 ", text)
    else:
        raise ValueError(f"不支持的语言: {lang}")

    return _SPACE_RE.sub(" ", text).strip()


def add_target_tokens(text: str) -> str:
    return f"[start] {text} [end]"


def detokenize_zh(tokens: list[str]) -> str:
    cleaned = [t for t in tokens if t not in {"", "[UNK]", "[start]", "[end]"}]
    text = "".join(cleaned)
    text = re.sub(r"\s+", "", text)
    return text.strip()

