"""Shared Ctrl+C cancellation for streaming LLM requests.

The streaming clients block inside ``response.iter_lines()`` on a socket
read. On Windows the default SIGINT handler does not reliably break that
blocking recv() until the next chunk arrives, so Ctrl+C can appear
unresponsive during long first-token waits. Closing the underlying
response from the signal handler forces ``iter_lines`` to raise
immediately, giving the user instant cancellation.

Second Ctrl+C restores the default handler so the user can hard-abort
even if a client is stuck outside a registered response.
"""

import signal
import threading
from contextlib import contextmanager

from ..cli_display import log

_cancel_event = threading.Event()
_active_response = None
_active_lock = threading.Lock()
_handler_installed = False
_previous_handler = None


def install_sigint_handler() -> None:
    """Install a SIGINT handler that cancels active streams on first press
    and restores the default handler so a second press hard-aborts."""
    global _handler_installed, _previous_handler
    if _handler_installed:
        return
    _previous_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, _handle_sigint)
    _handler_installed = True


def _handle_sigint(signum, frame):
    _cancel_event.set()
    with _active_lock:
        resp = _active_response
    if resp is not None:
        try:
            resp.close()
        except Exception:
            pass
    # Restore previous handler so a second Ctrl+C hard-aborts even if
    # we're stuck somewhere outside a registered stream.
    if _previous_handler is not None:
        signal.signal(signal.SIGINT, _previous_handler)
    log.warning("[cancel] Ctrl+C received — cancelling stream (press again to force-quit)")
    raise KeyboardInterrupt()


def check_cancelled() -> None:
    """Raise KeyboardInterrupt if cancellation was requested."""
    if _cancel_event.is_set():
        raise KeyboardInterrupt()


@contextmanager
def streaming_response(response):
    """Register a streaming ``requests.Response`` so the SIGINT handler
    can close its socket to unblock ``iter_lines``."""
    global _active_response
    with _active_lock:
        _active_response = response
    try:
        yield response
    finally:
        with _active_lock:
            if _active_response is response:
                _active_response = None
