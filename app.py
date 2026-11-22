import os
import sys
import threading
import time
import webbrowser
from pathlib import Path

SRC_DIR = str(Path(__file__).resolve().parent / "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

try:
    import spaces

    @spaces.GPU
    def _hf_zerogpu_heartbeat() -> str:
        return "DepShield AI ZeroGPU runtime active"
except ImportError:
    pass

import uvicorn

from api import app
from depshield.rag_agent import ThreatRAGAgent


def _ensure_vectorstore() -> None:
    try:
        ThreatRAGAgent().build_vector_index(force_rebuild=False)
    except Exception as exc:
        print(f"[DepShield] Vectorstore init notice: {exc}")


def _open_browser() -> None:
    time.sleep(1.2)
    webbrowser.open("http://127.0.0.1:7860")


if __name__ == "__main__":
    _ensure_vectorstore()
    if os.name == "nt":
        threading.Thread(target=_open_browser, daemon=True).start()
        uvicorn.run(app, host="127.0.0.1", port=7860)
    else:
        uvicorn.run(app, host="0.0.0.0", port=7860)