from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.getenv("RAG_DATA_DIR", str(ROOT / "rag" / "data")))
os.environ.setdefault("HF_HOME", str(DATA / "hf"))
_LOCAL_ENV = Path(__file__).resolve().parent / ".env.local"
if _LOCAL_ENV.is_file() and not os.getenv("DEEPSEEK_API_KEY"):
    for _line in _LOCAL_ENV.read_text(encoding="utf-8").splitlines():
        _name, _sep, _value = _line.partition("=")
        if _name.strip() == "DEEPSEEK_API_KEY" and _sep:
            _value = _value.strip()
            if len(_value) >= 2 and _value[0] == _value[-1] and _value[0] in "\"'":
                _value = _value[1:-1]
            if _value:
                os.environ["DEEPSEEK_API_KEY"] = _value
            break
PROCESSED = ROOT / "docling_extraction" / "processed"
RAW = ROOT / "docling_extraction" / "raw"
PDFS = ROOT / "textbooks"
CHUNK_VERSION = "heading-v6-pages-v1"
EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "BAAI/bge-large-en-v1.5")
RERANK_MODEL = os.getenv("RAG_RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
COLLECTION = "dentist_v1"
MAX_TOKENS = 512
SOFT_TOKENS = 400
CONTEXT_TOKENS = 7000
DEEPSEEK_MODEL = "deepseek-flash"
