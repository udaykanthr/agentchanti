import argparse
from .config import Config
from .llm.ollama import OllamaClient
from .llm.lm_studio import LMStudioClient
from .agents.planner import PlannerAgent
from .agents.coder import CoderAgent
from .agents.reviewer import ReviewerAgent

def main():
    parser = argparse.ArgumentParser(description="Multi-Agent Coder")
    parser.add_argument("task", help="The coding task to perform")
    parser.add_argument("--provider", choices=["ollama", "lm_studio"], default="ollama", help="The LLM provider to use")
    parser.add_argument("--model", default=Config.DEFAULT_MODEL, help="The model name to use")
    
    args = parser.parse_args()

    if args.provider == "ollama":
        llm_client = OllamaClient(base_url=Config.OLLAMA_BASE_URL, model=args.model)
    else:
        llm_client = LMStudioClient(base_url=Config.LM_STUDIO_BASE_URL, model=args.model)

    planner = PlannerAgent("Planner", "Senior Software Architect", "Create a detailed step-by-step plan for the coding task.", llm_client)
    coder = CoderAgent("Coder", "Senior Python Developer", "Write clean, efficient, and error-free Python code based on the plan.", llm_client)
    reviewer = ReviewerAgent("Reviewer", "Code Reviewer", "Review the code for logical errors, bugs, and style issues.", llm_client)

    print(f"--- Starting Task: {args.task} ---\n")

    print(f"--- {planner.name} ({planner.role}) ---\n")
    plan = planner.process(args.task)
    print(plan)
    print("\n--------------------------------------------------\n")

    print(f"--- {coder.name} ({coder.role}) ---\n")
    code = coder.process(f"Implement the following plan:\n{plan}")
    print(code)
    print("\n--------------------------------------------------\n")

    print(f"--- {reviewer.name} ({reviewer.role}) ---\n")
    review = reviewer.process(f"Review the following code:\n{code}", context=f"Original task: {args.task}")
    print(review)
    print("\n--------------------------------------------------\n")

if __name__ == "__main__":
    main()
