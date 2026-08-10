"""A finished run must not raise from rich's stream proxy during shutdown.

Observed at the end of a *successful* run, after the report path and
"Committed!"::

    Exception ignored in: <rich.file_proxy.FileProxy object at 0x...>
      File ".../rich/file_proxy.py", line 53, in flush
      File ".../rich/console.py", line 1705, in print
      File ".../rich/protocol.py", line 27, in rich_cast
    ImportError: sys.meta_path is None, Python is likely shutting down

In a terminal `Live.start()` swaps sys.stdout/sys.stderr for
`rich.file_proxy.FileProxy`, which buffers any write not ending in a
newline — a progress bar's ``\\r`` updates, for example. FileProxy extends
io.TextIOBase, so io.IOBase.__del__ closes and therefore FLUSHES it at
interpreter shutdown, where console.print reaches rich_cast's
function-level `from rich.console import RenderableType`. With sys.meta_path
torn down, that import raises.

THE FIRST FIX WAS WRONG and these tests encode why. Draining the current
sys.stdout/sys.stderr is not enough: each Live.start() mints a fresh proxy
pair and detaches the old one, while anything that captured a stream earlier
keeps writing to the DETACHED object (tqdm stores the file handed to it at
construction, so the KB embedder's "Embedding symbols" bar outlives several
pause/resume cycles). Every proxy ever installed must be drained, and it must
happen at atexit too, because a proxy can be re-dirtied after finish().
"""

import atexit
import gc
import io
import sys
import unittest
from contextlib import redirect_stdout
from unittest.mock import MagicMock, patch

from agentchanti.cli_display import CLIDisplay


def _proxy_with_buffered_text(text="Embedding symbols: 0batch"):
    """A real rich FileProxy holding an unterminated line."""
    from rich.console import Console
    from rich.file_proxy import FileProxy
    sink = io.StringIO()
    proxy = FileProxy(Console(file=sink, force_terminal=False, width=80), sink)
    proxy.write(text)                      # no trailing newline -> buffered
    assert sink.getvalue() == "", "precondition: text is still buffered"
    return proxy, sink


def _buffer_of(proxy):
    return "".join(getattr(proxy, "_FileProxy__buffer", []))


class DetachedProxiesAreDrainedTest(unittest.TestCase):
    """The regression that defeated the first fix."""

    def test_a_tracked_proxy_is_drained_even_when_not_the_current_stream(self):
        display = CLIDisplay("t")
        proxy, sink = _proxy_with_buffered_text()
        display._live_proxies = [proxy]     # detached: never sys.stdout/stderr

        display._drain_live_proxies()

        self.assertEqual(_buffer_of(proxy), "", "detached proxy left dirty")
        self.assertIn("Embedding symbols", sink.getvalue(),
                      "buffered text must reach the user, not the finalizer")

    def test_after_draining_the_finalizer_does_nothing(self):
        """The actual guarantee: an empty buffer cannot call console.print."""
        display = CLIDisplay("t")
        proxy, sink = _proxy_with_buffered_text()
        display._live_proxies = [proxy]
        display._drain_live_proxies()
        settled = sink.getvalue()

        del proxy
        display._live_proxies.clear()
        gc.collect()                        # run io.IOBase.__del__ now

        self.assertEqual(sink.getvalue(), settled,
                         "finalizer still emitted — buffer was not drained")

    def test_current_streams_are_drained_too(self):
        display = CLIDisplay("t")
        proxy, sink = _proxy_with_buffered_text("partial")
        real_out = sys.stdout
        sys.stdout = proxy
        try:
            display._drain_live_proxies()
        finally:
            sys.stdout = real_out
        self.assertEqual(_buffer_of(proxy), "")


class TrackingTest(unittest.TestCase):
    def test_only_rich_proxies_are_tracked(self):
        display = CLIDisplay("t")
        display._live_proxies = []
        with patch.object(sys, "stdout", io.StringIO()), \
             patch.object(sys, "stderr", io.StringIO()):
            display._track_live_proxies()
        self.assertEqual(display._live_proxies, [],
                         "plain streams are not proxies and need no tracking")

    def test_a_proxy_is_tracked_once(self):
        display = CLIDisplay("t")
        display._live_proxies = []
        proxy, _ = _proxy_with_buffered_text()
        with patch.object(sys, "stdout", proxy), \
             patch.object(sys, "stderr", proxy):
            display._track_live_proxies()
            display._track_live_proxies()
        self.assertEqual(len(display._live_proxies), 1)

    def test_construction_registers_an_atexit_drain(self):
        """A proxy can be re-dirtied after finish(); atexit is the backstop."""
        with patch.object(atexit, "register") as reg:
            display = CLIDisplay("t")
        registered = [c.args[0] for c in reg.call_args_list if c.args]
        self.assertTrue(
            any(getattr(f, "__func__", None) is CLIDisplay._drain_live_proxies
                or getattr(f, "__self__", None) is display
                for f in registered),
            f"drain not registered with atexit; got {registered}")


class DrainedBeforeTheProxyIsDroppedTest(unittest.TestCase):
    """Ordering: after Live.stop() the proxy is no longer the live stream."""

    def _display_recording_order(self):
        display = CLIDisplay("benchmark task")
        order = []
        display._live = MagicMock()
        display._live.stop.side_effect = lambda: order.append("stop")
        return display, order

    def test_finish_drains_before_stopping_live(self):
        display, order = self._display_recording_order()
        with patch.object(CLIDisplay, "_drain_live_proxies",
                          lambda self: order.append("drain")):
            with redirect_stdout(io.StringIO()):
                display.finish(success=True)
        self.assertEqual(order[:2], ["drain", "stop"])

    def test_pause_drains_before_stopping_live(self):
        display, order = self._display_recording_order()
        with patch.object(CLIDisplay, "_drain_live_proxies",
                          lambda self: order.append("drain")):
            with redirect_stdout(io.StringIO()):
                display.pause()
        self.assertEqual(order[:2], ["drain", "stop"])


class RobustnessTest(unittest.TestCase):
    def test_a_stream_that_raises_is_swallowed(self):
        """Teardown cosmetics must never break an otherwise finished run."""
        display = CLIDisplay("t")
        boom = MagicMock()
        boom.flush.side_effect = ValueError("I/O operation on closed file")
        display._live_proxies = [boom]
        display._drain_live_proxies()       # must not raise
        boom.flush.assert_called_once()

    def test_plain_streams_are_tolerated(self):
        """No Live, or not a terminal: no proxy was ever installed."""
        display = CLIDisplay("t")
        display._live_proxies = []
        display._drain_live_proxies()       # must not raise


if __name__ == "__main__":
    unittest.main()
