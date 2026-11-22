import os
import sys
import threading
import time
import webbrowser
from pathlib import Path

SRC_DIR = str(Path(__file__).resolve().parent / "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import uvicorn

from api import app


def _open_browser() -> None:
    time.sleep(1.2)
    webbrowser.open("http://127.0.0.1:7860")


if __name__ == "__main__":
    if os.name == "nt":
        threading.Thread(target=_open_browser, daemon=True).start()
        uvicorn.run(app, host="127.0.0.1", port=7860)
    else:
        uvicorn.run(app, host="0.0.0.0", port=7860)
