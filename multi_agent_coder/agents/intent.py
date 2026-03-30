import logging
import re
from .base import Agent

_logger = logging.getLogger(__name__)

class IntentAgent(Agent):
    """
    Analyzes the raw user prompt and gathers necessary context before planning.
    It can iteratively search the web or the local knowledge base if it
    determines that the prompt lacks sufficient technical clarity.

    IMPORTANT: This agent produces a SHORT intent clarification, NOT a full
    implementation guide.  The KB docs are passed separately to the planner
    and must remain the authoritative source for specific syntax, filenames,
    and configuration details.

    Supported commands the LLM can emit:
      SEARCH: <query>      — delegates to SearchAgent (web search)
      KB_SEARCH: <query>   — delegates to KB ContextBuilder (local codebase)
    """

    def process(self, task: str, context: str = "") -> str:
        """Implementation of base Agent process. Fallback if called directly."""
        return self.analyze_intent(task)

    def analyze_intent(
        self,
        raw_task: str,
        search_agent=None,
        kb_context_builder=None,
        kb_context: str = "",
        max_iterations: int = 3,
    ) -> str:
        """
        Analyze the intent of the raw task.  If the intent is unclear or
        requires specific technical documentation, the LLM can request:
          - SEARCH: <query>      → web search via SearchAgent
          - KB_SEARCH: <query>   → local file/KB search via ContextBuilder

        Returns the original task with a short INTENT_CLARIFICATION appended.
        """
        _logger.info("[IntentAnalysis] Starting intent analysis for task.")

        current_task = raw_task
        accumulated_context = ""
        if kb_context:
            accumulated_context += f"Project Stack / Knowledge Base Context:\n{kb_context}\n\n"

        for iteration in range(1, max_iterations + 1):
            prompt = (
                f"Role: {self.role}\n"
                f"Goal: {self.goal}\n\n"
                f"You are a requirements analyst evaluating a user prompt BEFORE it reaches "
                f"the planning agent.  Your job is to CLARIFY INTENT, not to write an "
                f"implementation guide.\n\n"
                f"User Prompt:\n{current_task}\n\n"
            )

            if accumulated_context:
                prompt += f"Gathered Context so far:\n{accumulated_context}\n\n"

            prompt += (
                "Instructions:\n"
                "You have access to two search tools:\n\n"
                "1. To search the WEB for latest docs, API references, or package info, output exactly:\n"
                "   SEARCH: <your web query>\n\n"
                "2. To search the LOCAL CODEBASE / knowledge base for existing files, components, "
                "or patterns in this project, output exactly:\n"
                "   KB_SEARCH: <your local query>\n\n"
                "You may only issue ONE command per response.\n\n"
                "If the prompt is clear and you have enough context, output a SHORT intent "
                "clarification starting with:\n"
                "REQUIREMENTS_SPEC:\n\n"
                "CRITICAL RULES for the REQUIREMENTS_SPEC:\n"
                "- Keep it CONCISE — at most 15-20 lines.\n"
                "- State ONLY: (1) the user's goal in one sentence, (2) key constraints or "
                "ambiguities you resolved, (3) any specific versions or packages identified.\n"
                "- Do NOT paraphrase, summarize, or re-explain KB documentation content. "
                "The KB docs will be passed separately to the planner.\n"
                "- Do NOT generate implementation guides, file structures, code snippets, "
                "acceptance criteria, or step-by-step instructions.\n"
                "- Do NOT duplicate information already present in the KB docs.\n"
                "- Focus on what the USER WANTS, not HOW to build it.\n"
            )

            try:
                response = self.llm_client.generate_response(prompt)

                # ── Check for KB_SEARCH command ──
                kb_match = re.search(
                    r'^KB_SEARCH:\s*(.+)$', response,
                    re.MULTILINE | re.IGNORECASE,
                )
                if kb_match and kb_context_builder is not None:
                    query = kb_match.group(1).strip()
                    _logger.info(
                        "[IntentAnalysis] Iteration %d: Requested KB search for: %s",
                        iteration, query,
                    )
                    kb_result = self._query_kb(kb_context_builder, query)
                    if kb_result:
                        accumulated_context += (
                            f"KB Search Results for '{query}':\n{kb_result}\n\n"
                        )
                        continue
                    else:
                        _logger.warning(
                            "[IntentAnalysis] KB search returned no results for: %s",
                            query,
                        )

                # ── Check for web SEARCH command ──
                search_match = re.search(
                    r'^SEARCH:\s*(.+)$', response,
                    re.MULTILINE | re.IGNORECASE,
                )
                if search_match and search_agent is not None:
                    query = search_match.group(1).strip()
                    _logger.info(
                        "[IntentAnalysis] Iteration %d: Requested web search for: %s",
                        iteration, query,
                    )
                    search_result = search_agent.search_for_task(
                        query, kb_context=kb_context,
                    )
                    if search_result:
                        accumulated_context += (
                            f"Web Search Results for '{query}':\n{search_result}\n\n"
                        )
                        continue
                    else:
                        _logger.warning(
                            "[IntentAnalysis] Web search returned no results for: %s",
                            query,
                        )

                # ── Check for REQUIREMENTS_SPEC ──
                spec_match = re.search(
                    r'REQUIREMENTS_SPEC:(.+)', response,
                    re.IGNORECASE | re.DOTALL,
                )
                if spec_match:
                    spec = spec_match.group(1).strip()
                    _logger.info(
                        "[IntentAnalysis] Successfully generated REQUIREMENTS_SPEC.",
                    )
                    # Append ONLY the concise clarification to the original task.
                    # Do NOT include accumulated_context here — the KB docs and
                    # search results were only needed for the IntentAgent's LLM
                    # to make its decision.  The planner gets KB docs through
                    # its own channel (step 4 of pre_analyze), so including them
                    # here would cause massive context duplication and pollute
                    # downstream KB search queries.
                    return (
                        f"{raw_task}\n\n"
                        f"=== INTENT CLARIFICATION (from requirements analyst) ===\n"
                        f"{spec}"
                    )

                # If neither matched exactly, treat the response as the spec
                _logger.info(
                    "[IntentAnalysis] No exact marker found, treating response as spec.",
                )
                return (
                    f"{raw_task}\n\n"
                    f"=== INTENT CLARIFICATION (from requirements analyst) ===\n"
                    f"{response.strip()}"
                )

            except Exception as e:
                _logger.warning(
                    "[IntentAnalysis] Failed during iteration %d: %s", iteration, e,
                )
                break

        # Fallback to the original task if all iterations fail
        _logger.warning(
            "[IntentAnalysis] Returning original raw task as fallback.",
        )
        return raw_task

    # ── KB search helper ─────────────────────────────────────────

    @staticmethod
    def _query_kb(kb_context_builder, query: str) -> str:
        """Query the global KB store for docs/patterns matching *query*.

        Returns a formatted string of matching documents, or empty string
        if nothing was found or the KB is unavailable.
        """
        try:
            kb_context_builder._ensure_global()
            gs = kb_context_builder._global_store
            if gs is None:
                return ""

            hits = gs.search(
                query=query,
                categories=["doc", "pattern"],
                top_k=5,
                api_client=kb_context_builder._api_client,
            )
            if not hits:
                return ""

            sections = []
            for h in hits:
                sections.append(f"### {h.title}\n{h.content or ''}")
            return "\n\n".join(sections)

        except Exception as exc:
            _logger.debug("[IntentAnalysis] KB query failed: %s", exc)
            return ""
