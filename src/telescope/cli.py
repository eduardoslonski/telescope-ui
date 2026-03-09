"""Telescope CLI — start the visualization server."""

from __future__ import annotations

import argparse
import logging
import platform
import subprocess
import sys
import threading
import time


BLUE = "\033[34m"
RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[1;32m"
DIM = "\033[90m"
UNDER = "\033[1;4m"
CLEAR_LINE = "\033[2K\033[G"

_LOGO = f"""{BLUE}
  _______   _
 |__   __| | |
    | | ___| | ___  ___  ___ ___  _ __   ___
    | |/ _ \\ |/ _ \\/ __|/ __/ _ \\| '_ \\ / _ \\
    | |  __/ |  __/\\__ \\ (_| (_) | |_) |  __/
    |_|\\___|_|\\___||___/\\___\\___/| .__/ \\___|
                                 | |
                                 |_|         {RESET}"""


def _open_browser(url: str) -> None:
    """Open *url* in the default browser after a short delay."""
    time.sleep(1.5)
    system = platform.system()
    try:
        if system == "Darwin":
            subprocess.Popen(["open", url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elif system == "Linux":
            subprocess.Popen(["xdg-open", url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            import webbrowser
            webbrowser.open(url)
    except Exception:
        pass


def main() -> None:
    # Print logo + "Starting..." immediately, before any heavy imports
    sys.stderr.write(_LOGO + "\n")
    sys.stderr.write(f"  {DIM}Starting...{RESET}")
    sys.stderr.flush()

    parser = argparse.ArgumentParser(
        prog="telescope",
        description="Telescope \u2014 LLM training visualization",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8005,
        help="Port to serve on (default: 8005)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't open browser automatically",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Run in development mode (auto-reload, no static serving)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging (verbose W&B sync details, DB operations)",
    )
    args = parser.parse_args()

    # Heavy imports happen here
    import uvicorn

    from telescope import __version__
    from telescope.server.db import _get_data_dir

    data_dir = _get_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)

    if not args.dev:
        from telescope.server.main import mount_static_ui
        mount_static_ui()

    url = f"http://{args.host}:{args.port}"

    # Replace "Starting..." with the full info block
    sys.stderr.write(CLEAR_LINE)
    sys.stderr.write(
        f"  {BOLD}v{__version__}{RESET}\n"
        f"\n"
        f"  {GREEN}>{RESET}  Ready at {UNDER}{url}{RESET}\n"
        f"  {DIM}>{RESET}  Data dir {DIM}{data_dir}{RESET}\n"
        f"  {DIM}>{RESET}  Press {BOLD}Ctrl+C{RESET} to stop\n\n"
    )
    sys.stderr.flush()

    # Logging: only show logs in --debug mode
    if args.debug:
        log_level = logging.DEBUG
        uv_log_level = "debug"
    else:
        log_level = logging.WARNING
        uv_log_level = "warning"

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger().setLevel(log_level)

    if not args.no_browser:
        threading.Thread(target=_open_browser, args=(url,), daemon=True).start()

    try:
        uvicorn.run(
            "telescope.server.main:app",
            host=args.host,
            port=args.port,
            reload=args.dev,
            log_level=uv_log_level,
        )
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
