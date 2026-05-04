"""Entry point — run the trading signal scheduler.

Usage:
    # Foreground (logs appear in the terminal):
    .\\venv\\Scripts\\python run_scheduler.py

    # Background (Windows — logs saved to scheduler.log):
    start /B .\\venv\\Scripts\\python run_scheduler.py > scheduler.log 2>&1

    # Stop background process:
    taskkill /F /IM python.exe   (use only if no other Python processes are running)

Prerequisites:
    1. Copy .env.example to .env and fill in TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID
    2. Run `python main.py` at least once to populate the SQLite database
"""

from __future__ import annotations

import logging
import os
import sys

# Force UTF-8 stdout so emoji in log messages render correctly on Windows
os.environ.setdefault("PYTHONUTF8", "1")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Load .env before importing anything else so config picks it up
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)

from core.scheduler import run_scheduler  # noqa: E402 (after basicConfig)

if __name__ == "__main__":
    run_scheduler()
