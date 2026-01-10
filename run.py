from __future__ import annotations

import torch

from app.ui import build_demo
from app.utils import load_css


if __name__ == "__main__":
    dev_auto = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[app] Auto device = {dev_auto}")
    demo = build_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        css=load_css(),
        )
