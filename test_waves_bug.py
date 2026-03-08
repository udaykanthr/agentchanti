import logging
from multi_agent_coder.executor import Executor
from multi_agent_coder.orchestrator.plan_optimizer import optimize_plan
from multi_agent_coder.orchestrator.pipeline import build_step_waves

# Setup minimal logging to see what happens
logging.basicConfig(level=logging.DEBUG)

plan_text = """1. Initialize the Vite React project in a new `single-page-responsive-web-application` folder using the React template with `npm create vite@latest single-page-responsive-web-application -- --template react --no-interactive` (depends: none)

2. Install React Router dependency in `single-page-responsive-web-application` with `cd single-page-responsive-web-application && npm install react-router-dom` (depends: 1)

3. Install Tailwind CSS v4 and PostCSS dependencies in `single-page-responsive-web-application` with `cd single-page-responsive-web-application && npm install tailwindcss @tailwindcss/postcss postcss` (depends: 1)

4. Configure PostCSS to use Tailwind by creating or updating `single-page-responsive-web-application/postcss.config.mjs` to export a default config object with `plugins` containing `"@tailwindcss/postcss": {}` (depends: 3)

5. Enable Tailwind in the main stylesheet by updating `single-page-responsive-web-application/src/index.css` (or `src/main.css` if present instead) to remove any old Tailwind v3 `@tailwind base;`, `@tailwind components;`, and `@tailwind utilities;` directives and replace them with a single `@import "tailwindcss";` at the top, plus any minimal custom global styles if desired (depends: 3)

6. Ensure the React entry point imports the Tailwind-enabled stylesheet by updating `single-page-responsive-web-application/src/main.jsx` to keep the ReactDOM rendering logic, import `React`, `ReactDOM`, `App` from `./App.jsx`, and import the updated CSS file (`./index.css` or `./main.css`), without introducing TypeScript (depends: 1, 5)

7. Implement the application layout and routing by updating `single-page-responsive-web-application/src/App.jsx` to:
   - Import `BrowserRouter`, `Routes`, and `Route` from `react-router-dom`
   - Import the layout components `Header`, `HeroBanner`, `PriceList`, `CardSection`, and `LargeFooter`
   - Define a single-page layout in the render tree with components ordered as: `Header`, `HeroBanner`, `PriceList`, `CardSection`, `LargeFooter`
   - Wrap the layout in `BrowserRouter` (even if only one route is used) to allow future navigation
   - Apply Tailwind utility classes to create a responsive single-page layout (e.g., container widths, spacing, background colors) (depends: 2, 5)
"""

print("--- 1. PARSE PLAN ---")
raw_steps = Executor.parse_plan_steps(plan_text)
print(f"Num raw steps: {len(raw_steps)}")
for i, s in enumerate(raw_steps):
    print(f"[{i}]: {s}")

print("\n--- 2. PARSE DEPS ---")
steps, deps = Executor.parse_step_dependencies(raw_steps)
for i in range(len(steps)):
    print(f"deps[{i}] = {deps[i]}")

print("\n--- 3. OPTIMIZE PLAN ---")
opt_steps, opt_deps = optimize_plan(steps, dependencies=deps)
for i in range(len(opt_steps)):
    print(f"opt_deps[{i}] = {opt_deps.get(i, set())}")

print("\n--- 4. BUILD WAVES ---")
waves = build_step_waves(opt_steps, opt_deps)
print(f"Waves: {waves}")
