import pytest
from multi_agent_coder.diff_display import _detect_hazards, HAZARD_WARN
from multi_agent_coder.config import Config

def test_detect_hazards_shrinkage():
    # File shrinkage > 50%
    old_content = "a" * 200
    new_content = "a" * 50
    hazards = _detect_hazards("any_file.txt", old_content, new_content)
    assert len(hazards) == 1
    assert hazards[0][0] == HAZARD_WARN
    assert "Significant size reduction" in hazards[0][1]

def test_detect_hazards_no_shrinkage():
    # Small shrinkage is fine
    old_content = "a" * 200
    new_content = "a" * 150
    hazards = _detect_hazards("any_file.txt", old_content, new_content)
    assert len(hazards) == 0

def test_detect_hazards_small_file_ignored():
    # Small files ignored
    old_content = "a" * 50
    new_content = "a" * 10
    hazards = _detect_hazards("any_file.txt", old_content, new_content)
    assert len(hazards) == 0

# def test_detect_hazards_package_json_dependency_loss():
#     # Dependencies removed
#     old_content = '{"dependencies": {"react": "^18.0.0"}}'
#     new_content = '{"name": "my-app"}'
#     hazards = _detect_hazards("package.json", old_content, new_content)
#     # Checks for dependency loss heuristics
#     assert any("Critical section 'dependencies' removed" in h[1] for h in hazards)

def test_detect_hazards_package_json_additive_is_safe():
    # Adding a NEW dependency is safe — only version CHANGES are flagged
    old_content = '{"name": "app", "dependencies": {}}'
    new_content = '{"name": "app", "dependencies": {"react": "^18"}}'
    
    hazards = _detect_hazards("package.json", old_content, new_content)
    assert len(hazards) == 0

def test_detect_hazards_package_json_version_change_flagged():
    # Changing an existing dependency version IS flagged
    old_content = '{"name": "app", "dependencies": {"react": "^19.0.0"}}'
    new_content = '{"name": "app", "dependencies": {"react": "^18.3.1"}}'
    
    hazards = _detect_hazards("package.json", old_content, new_content)
    assert any("Dependency version changes" in h[1] for h in hazards)

def test_detect_hazards_package_json_safe_edit():
    # Editing scripts only — no dependency warning
    old_content = '{"scripts": {"test": "jest"}}'
    new_content = '{"scripts": {"test": "vitest"}}'
    hazards = _detect_hazards("package.json", old_content, new_content)
    assert len(hazards) == 0
