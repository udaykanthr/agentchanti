"""The evidence pre-flight and the seeder must agree on which languages
the seeder serves.

`cli.py` decides whether to warn "nothing can satisfy
`require_independent_evidence` in this run" from whether the seeder can
write a contract. It restated that as ``("python", "py")`` and had
already drifted twice — past `_seed_js` when JavaScript gained a
contract, and past `_seed_go_builtin` when Go did.

The drift is latent rather than loud: the seeder runs before the
pre-flight, so a successful seed populates the pre-existing set and the
branch is skipped. It surfaces only when seeding FAILS, and then it
misdirects — telling a JavaScript or Go user that the seeder "does not
support" their language and is "Python only", neither of which is true.

So the predicate is asked of the seeder, and this test pins that the two
cannot disagree again.
"""
import pytest

from agentchanti.orchestrator.acceptance_seed import (seedable_language,
                                                      seedable_languages_note)


class TestLanguagesTheSeederServes:

    @pytest.mark.parametrize("lang", [
        "python", "py", "Python",
        "javascript", "js", "typescript", "ts", "node", "jsx", "tsx",
        "go", "golang", "Go",
        "rust", "rs", "Rust",
        "java", "Java",
    ])
    def test_a_served_language_is_seedable(self, lang):
        assert seedable_language(lang)

    @pytest.mark.parametrize("lang", ["c", "cpp", "ruby",
                                      "php", "csharp"])
    def test_an_unserved_language_is_not(self, lang):
        """Go, Rust and Java each moved to the served list on gaining a
        contract; move the next one along rather than dropping the case."""
        assert not seedable_language(lang)

    def test_unknown_language_is_treated_as_seedable(self):
        """Detection has not run or could not decide, and the seeder falls
        back to the Python path. Claiming the flag is unsatisfiable on a
        guess would be worse than saying nothing."""
        assert seedable_language(None)
        assert seedable_language("")


class TestItMatchesTheSeederItself:
    """A hardcoded list is exactly what drifted; these assert the
    predicate is derived from the same sets the dispatch uses."""

    def test_every_js_language_in_the_dispatch_set(self):
        from agentchanti.orchestrator.acceptance_seed import _JS_LANGUAGES
        for lang in _JS_LANGUAGES:
            assert seedable_language(lang), lang

    def test_every_go_language_in_the_dispatch_set(self):
        from agentchanti.orchestrator.acceptance_seed import _GO_LANGUAGES
        for lang in _GO_LANGUAGES:
            assert seedable_language(lang), lang

    def test_every_rust_language_in_the_dispatch_set(self):
        from agentchanti.orchestrator.acceptance_seed import _RUST_LANGUAGES
        for lang in _RUST_LANGUAGES:
            assert seedable_language(lang), lang

    def test_every_java_language_in_the_dispatch_set(self):
        from agentchanti.orchestrator.acceptance_seed import _JAVA_LANGUAGES
        for lang in _JAVA_LANGUAGES:
            assert seedable_language(lang), lang

    def test_the_note_names_what_is_supported(self):
        note = seedable_languages_note()
        for word in ("Python", "JavaScript", "Go", "Rust", "Java"):
            assert word in note


class TestTheCliConsumesIt:
    """A predicate nothing calls is the mistake `protect_acceptance_files`
    already made once."""

    def test_cli_asks_the_seeder_rather_than_restating_it(self):
        import inspect

        from agentchanti.orchestrator import cli
        src = inspect.getsource(cli)
        assert "seedable_language(language)" in src
        assert 'language.lower() in ("python", "py")' not in src, (
            "the pre-flight is restating the seeder's capability again")
