import os
import sys
import threading
import time
import webbrowser
from pathlib import Path

try:
    import spaces

    @spaces.GPU
    def _hf_zerogpu_heartbeat() -> str:
        return "DepShield AI ZeroGPU runtime active"
except ImportError:
    spaces = None

import torch

_orig_cuda_available = torch.cuda.is_available
torch.cuda.is_available = lambda: False

SRC_DIR = str(Path(__file__).resolve().parent / "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import uvicorn
from api import app
from depshield.rag_agent import ThreatRAGAgent

torch.cuda.is_available = _orig_cuda_available


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
        import gradio as gr

        with gr.Blocks() as demo:
            btn = gr.Button("Heartbeat")
            if spaces is not None:
                btn.click(fn=_hf_zerogpu_heartbeat, outputs=gr.Textbox())

        demo.launch(server_name="0.0.0.0", server_port=7860, prevent_thread_lock=True)
        demo.app.router.routes[0:0] = app.router.routes
        while True:
            time.sleep(3600)