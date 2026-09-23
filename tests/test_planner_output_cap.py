"""The planner gets a larger output ceiling than the other agents.

A truncated plan is thrown away and regenerated in full, so hitting the cap
while planning costs the whole generation twice (measured 2026-09-22 on
glm-5.3-flash: 2 of 3 runs re-planned after hitting 16,384).
"""
from agentchanti.config import Config


def _cfg(tmp_path, monkeypatch, yaml_text=""):
    import yaml
    monkeypatch.delenv("PLANNER_MAX_OUTPUT_TOKENS", raising=False)
    monkeypatch.delenv("MAX_OUTPUT_TOKENS", raising=False)
    return Config(yaml_data=yaml.safe_load(yaml_text) or {})


def test_default_is_larger_than_the_global_cap(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, monkeypatch)
    assert cfg.PLANNER_MAX_OUTPUT_TOKENS == 32768
    assert cfg.MAX_OUTPUT_TOKENS == 16384


def test_yaml_overrides_it(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, monkeypatch, "planner_max_output_tokens: 65536\n")
    assert cfg.PLANNER_MAX_OUTPUT_TOKENS == 65536


def test_never_below_the_global_cap(tmp_path, monkeypatch):
    """A smaller planner cap would make planning the likeliest to truncate."""
    cfg = _cfg(tmp_path, monkeypatch,
               "max_output_tokens: 50000\nplanner_max_output_tokens: 8000\n")
    assert cfg.PLANNER_MAX_OUTPUT_TOKENS == 50000
