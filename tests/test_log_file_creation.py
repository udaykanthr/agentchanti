"""A log file appears only when there is something to log.

`log = setup_logger()` runs at module scope, so ANY import of the package
opened a timestamped log — including short-lived subprocess utilities
that never log a line. Observed once the style-coupling gate began
running `python -m agentchanti...` on every check: a single run left ten
zero-byte files beside its real 47 KB log, cluttering exactly the
directory someone opens to find out what happened.
"""

import glob
import logging
import os

from agentchanti.cli_display import setup_logger


def _logs(root):
    return glob.glob(os.path.join(root, ".agentchanti", "logs", "*.log"))


class TestLazyLogFile:
    def test_setup_alone_creates_no_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        logger = setup_logger()
        handlers = list(logger.handlers)
        try:
            assert _logs(str(tmp_path)) == []
        finally:
            for h in handlers[-1:]:
                logger.removeHandler(h)
                h.close()

    def test_the_first_record_creates_it(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        logger = setup_logger()
        handler = logger.handlers[-1]
        try:
            logger.info("something worth recording")
            handler.flush()
            files = _logs(str(tmp_path))
            assert len(files) == 1
            assert os.path.getsize(files[0]) > 0
            assert "something worth recording" in open(
                files[0], encoding="utf-8").read()
        finally:
            logger.removeHandler(handler)
            handler.close()

    def test_the_handler_is_deferred(self, tmp_path, monkeypatch):
        """Pins the mechanism, not just today's symptom."""
        monkeypatch.chdir(tmp_path)
        logger = setup_logger()
        handler = logger.handlers[-1]
        try:
            assert isinstance(handler, logging.FileHandler)
            assert handler.delay, "FileHandler must defer opening its file"
            assert handler.stream is None
        finally:
            logger.removeHandler(handler)
            handler.close()
