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
from api import FRONTEND_HTML, app
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
        import gradio.routes
        from fastapi.responses import FileResponse

        _orig_create_app = gradio.routes.App.create_app

        def _patched_create_app(*args, **kwargs):
            gradio_app = _orig_create_app(*args, **kwargs)
            gradio_app.router.routes[0:0] = app.router.routes

            @gradio_app.middleware("http")
            async def _serve_depshield_frontend(request, call_next):
                if request.url.path == "/":
                    return FileResponse(FRONTEND_HTML)
                return await call_next(request)

            return gradio_app

        gradio.routes.App.create_app = _patched_create_app

        with gr.Blocks() as demo:
            btn = gr.Button("Heartbeat")
            if spaces is not None:
                btn.click(fn=_hf_zerogpu_heartbeat, outputs=gr.Textbox())

        demo.launch(server_name="0.0.0.0", server_port=7860, ssr_mode=False)