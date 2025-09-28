"""Main controller for hotkey-triggered audio transcription and HTML logging.

Run this module to keep a background listener active. Press the configured
hotkey from anywhere on the system to capture the last 30 seconds of system
output audio, send it to the LLM, and update the auto-refreshing HTML view.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable, List

from pynput import keyboard

from llm import prompt_llm
from zero_to_three import ensure_audio_capture, transcribe_last_30s

# ------------------------------ Configuration ---------------------------------

APP_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = APP_ROOT / "runtime"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HTML_PATH = OUTPUT_DIR / "LLM.html"
STATE_PATH = OUTPUT_DIR / "LLM_history.json"
LOG_PATH = OUTPUT_DIR / "LLM_log.txt"

HOTKEY = "<ctrl>+<alt>+l"  # Change to any pynput-compatible combination.
HTTP_HOST = "127.0.0.1"
HTTP_PORT = 8000
AUTO_REFRESH_SECONDS = 2

# ------------------------------- Data models ----------------------------------

@dataclass
class Exchange:
    timestamp: float
    prompt: str
    response: str

    @property
    def iso_timestamp(self) -> str:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.timestamp))

# ----------------------------- Persistence layer ------------------------------

def load_history() -> List[Exchange]:
    if not STATE_PATH.exists():
        return []
    try:
        with STATE_PATH.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        return [Exchange(**entry) for entry in raw]
    except Exception:
        # Fallback: start fresh if file is corrupt.
        return []


def save_history(history: List[Exchange]) -> None:
    serialised = [exchange.__dict__ for exchange in history]
    tmp_path = STATE_PATH.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8") as tmp:
        json.dump(serialised, tmp, ensure_ascii=False, indent=2)
    tmp_path.replace(STATE_PATH)


def append_text_log(exchange: Exchange) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as log:
        log.write("\n--- Transcription @ {ts} ---\n".format(ts=exchange.iso_timestamp))
        log.write(f"Prompt:\n{exchange.prompt}\n\n")
        log.write("--- LLM Response ---\n")
        log.write(f"{exchange.response}\n")
        log.write("-" * 48 + "\n")

# ------------------------------- HTML rendering -------------------------------

def render_html(history: List[Exchange]) -> str:
    entries = []
    for exchange in reversed(history):
        block = f"""
        <article class=\"exchange\">
            <header>
                <h2>{exchange.iso_timestamp}</h2>
            </header>
            <section>
                <h3>Prompt</h3>
                <p class=\"prompt\">{escape_html(exchange.prompt)}</p>
            </section>
            <section>
                <h3>Response</h3>
                <p class=\"response\">{escape_html(exchange.response)}</p>
            </section>
        </article>
        """
        entries.append(block)
    body = "\n".join(entries) or "<p>No exchanges captured yet. Press the hotkey to begin.</p>"
    return f"""
    <!DOCTYPE html>
    <html lang=\"en\">
    <head>
        <meta charset=\"UTF-8\">
        <meta http-equiv=\"refresh\" content=\"{AUTO_REFRESH_SECONDS}\">
        <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">
        <title>LLM Output Log</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 24px; background: #f5f5f5; }}
            h1 {{ margin-bottom: 16px; }}
            article {{ background: #fff; border-radius: 8px; padding: 16px; margin-bottom: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
            h2 {{ margin: 0 0 12px; font-size: 1.1rem; color: #333; }}
            h3 {{ margin: 12px 0 8px; font-size: 0.95rem; color: #555; text-transform: uppercase; letter-spacing: 0.03em; }}
            p {{ white-space: pre-wrap; color: #222; }}
            p.prompt {{ font-weight: 500; }}
            footer {{ margin-top: 24px; font-size: 0.85rem; color: #666; }}
            .hotkey {{ font-family: monospace; }}
        </style>
    </head>
    <body>
        <h1>Latest LLM Exchanges</h1>
        <p>Press <span class=\"hotkey\">{HOTKEY}</span> to capture the latest snippet.</p>
        {body}
        <footer>Auto-refreshes every {AUTO_REFRESH_SECONDS} seconds. Serving from http://{HTTP_HOST}:{HTTP_PORT}/LLM.html</footer>
    </body>
    </html>
    """


def escape_html(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def write_html(history: List[Exchange]) -> None:
    html = render_html(history)
    tmp_path = HTML_PATH.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8") as tmp:
        tmp.write(html)
    tmp_path.replace(HTML_PATH)

# ----------------------------- HTTP server helper -----------------------------

class QuietRequestHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:
        # Suppress request logging noise in the terminal.
        return


class StaticServer:
    def __init__(self, directory: Path, host: str, port: int):
        handler = partial(QuietRequestHandler, directory=str(directory))
        self._httpd = ThreadingHTTPServer((host, port), handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._httpd.shutdown()
        self._thread.join(timeout=2)

# ----------------------------- Hotkey management ------------------------------

class HotkeyProcessor:
    def __init__(self, hotkey: str, callback: Callable[[], None]):
        self._callback = callback
        self._listener = keyboard.GlobalHotKeys({hotkey: self._handle})
        self._thread = None

    def _handle(self) -> None:
        # Run callback on a worker thread to avoid blocking pynput internals.
        worker = threading.Thread(target=self._callback, daemon=True)
        worker.start()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._listener.start, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._listener.stop()
        if self._thread:
            self._thread.join(timeout=2)

# ---------------------------- Transcription pipeline --------------------------

class Pipeline:
    def __init__(self):
        ensure_audio_capture()
        self._history = load_history()
        self._lock = threading.Lock()
        write_html(self._history)

    def run_once(self) -> None:
        if not self._lock.acquire(blocking=False):
            print("Pipeline already running; ignoring hotkey press.")
            return
        try:
            print("\n--- Hotkey received: capturing audio ---")
            prompt = transcribe_last_30s().strip()
            if not prompt:
                prompt = "(no speech detected)"
            print(f"Prompt: {prompt}")

            print("--- Querying LLM ---")
            llm_response = prompt_llm(f"Explain this: {prompt}")
            print(f"Response: {llm_response}")

            exchange = Exchange(timestamp=time.time(), prompt=prompt, response=llm_response)
            self._history.append(exchange)
            save_history(self._history)
            write_html(self._history)
            append_text_log(exchange)

            print(f"Updated HTML at http://{HTTP_HOST}:{HTTP_PORT}/LLM.html")
        except Exception as exc:
            print(f"Pipeline failed: {exc}")
        finally:
            self._lock.release()

# --------------------------------- Entrypoint ---------------------------------

def main() -> None:
    pipeline = Pipeline()
    server = StaticServer(OUTPUT_DIR, HTTP_HOST, HTTP_PORT)
    server.start()
    hotkey = HotkeyProcessor(HOTKEY, pipeline.run_once)
    hotkey.start()

    print(
        "System ready. Hotkey {hk} will capture the last 30 seconds of audio.\n"
        "Open http://{host}:{port}/LLM.html in your browser to view live updates.".format(
            hk=HOTKEY, host=HTTP_HOST, port=HTTP_PORT
        )
    )

    try:
        # Keep the main thread alive while background threads handle work.
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        hotkey.stop()
        server.stop()


if __name__ == "__main__":
    main()
