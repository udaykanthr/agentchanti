"""
Seeder — populates the global knowledge base with sample data.

Seeds:
- errors.db with 5 errors per language (Python, JavaScript, TypeScript,
  Java, Go, Rust, C#)
- registry/ markdown files with frontmatter for patterns, ADRs, docs,
  and behavioral categories
- SQLite vector store with embedded markdown chunks

Designed as a dev-utility that can be re-run to reset sample data.
"""

from __future__ import annotations

import logging
import os
import uuid
from typing import Optional

from .error_dict import ErrorDict, ErrorFix, ContentFix

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_GLOBAL_DIR = os.path.dirname(os.path.abspath(__file__))
_CORE_DIR = os.path.join(_GLOBAL_DIR, "core")
_REGISTRY_DIR = os.path.join(_GLOBAL_DIR, "registry")


def _errors_db_path() -> str:
    return os.path.join(_CORE_DIR, "errors.db")


# ---------------------------------------------------------------------------
# Error seed data
# ---------------------------------------------------------------------------

_ERROR_SEEDS: list[ErrorFix] = [
    # ── Python ──────────────────────────────────────────────────────────
    ErrorFix(
        error_type="AttributeError",
        language="python",
        pattern=r"AttributeError:\s*'(\w+)'\s+object\s+has\s+no\s+attribute",
        cause="Accessing an attribute that does not exist on the object.",
        fix_template="Check the object type with type(obj) and verify the attribute name. "
                     "Use hasattr(obj, 'attr') before access or handle with getattr(obj, 'attr', default).",
        severity="error",
        tags="attribute,none,object,python",
    ),
    ErrorFix(
        error_type="ImportError",
        language="python",
        pattern=r"(ImportError|ModuleNotFoundError):\s*(No module named|cannot import name)",
        cause="The module or name is not installed or not in sys.path.",
        fix_template="Install the missing package with pip install <package>. "
                     "Check for typos in the import name. Verify the module is in your PYTHONPATH.",
        severity="error",
        tags="import,module,package,python",
    ),
    ErrorFix(
        error_type="TypeError",
        language="python",
        pattern=r"TypeError:\s*(unsupported operand|.+takes\s+\d+\s+positional|.+not\s+(callable|subscriptable|iterable))",
        cause="Operation applied to an object of inappropriate type.",
        fix_template="Inspect the types of all operands with type(). "
                     "Check function signatures match the call. Use isinstance() for type guards.",
        severity="error",
        tags="type,argument,callable,python",
    ),
    ErrorFix(
        error_type="KeyError",
        language="python",
        pattern=r"KeyError:\s*",
        cause="Dictionary key does not exist.",
        fix_template="Use dict.get(key, default) instead of dict[key]. "
                     "Or check with 'if key in dict:' before access.",
        severity="error",
        tags="key,dict,dictionary,missing,python",
    ),
    ErrorFix(
        error_type="RecursionError",
        language="python",
        pattern=r"RecursionError:\s*maximum recursion depth exceeded",
        cause="Infinite or excessively deep recursion.",
        fix_template="Add a proper base case to the recursive function. "
                     "Consider converting to an iterative approach. "
                     "If legitimate, use sys.setrecursionlimit() cautiously.",
        severity="error",
        tags="recursion,stack,overflow,depth,python",
    ),

    # ── JavaScript ──────────────────────────────────────────────────────
    ErrorFix(
        error_type="TypeError",
        language="javascript",
        pattern=r"TypeError:\s*(Cannot read propert|.+is not a function|.+is undefined|.+is null)",
        cause="Attempted to use undefined or null as an object.",
        fix_template="Add null/undefined checks: use optional chaining (obj?.prop) "
                     "and nullish coalescing (obj ?? default). Verify the variable is "
                     "initialized before use.",
        severity="error",
        tags="undefined,null,property,function,javascript",
    ),
    ErrorFix(
        error_type="ReferenceError",
        language="javascript",
        pattern=r"ReferenceError:\s*(\w+)\s+is\s+not\s+defined",
        cause="Variable or function referenced before declaration.",
        fix_template="Ensure the variable is declared with let/const/var before use. "
                     "Check for typos in the variable name. Verify the module is imported.",
        severity="error",
        tags="reference,undefined,variable,scope,javascript",
    ),
    ErrorFix(
        error_type="UnhandledPromiseRejection",
        language="javascript",
        pattern=r"(Unhandled\s*promise\s*rejection|UnhandledPromiseRejectionWarning)",
        cause="A Promise rejected without a .catch() handler or try/catch in async.",
        fix_template="Add .catch() to every promise chain, or wrap async calls in try/catch. "
                     "Add a global handler: process.on('unhandledRejection', handler).",
        severity="error",
        tags="promise,async,rejection,unhandled,javascript",
    ),
    ErrorFix(
        error_type="SyntaxError",
        language="javascript",
        pattern=r"SyntaxError:\s*(Unexpected token|Unexpected end of)",
        cause="Invalid JavaScript syntax — missing bracket, parenthesis, or semicolon.",
        fix_template="Check for missing closing brackets/parens. Verify JSON is valid. "
                     "Look for accidental use of reserved words.",
        severity="error",
        tags="syntax,token,parse,bracket,javascript",
    ),
    ErrorFix(
        error_type="RangeError",
        language="javascript",
        pattern=r"RangeError:\s*(Maximum call stack|Invalid array length)",
        cause="Value out of allowed range — often infinite recursion or invalid array size.",
        fix_template="Add a base case to recursive functions. "
                     "Validate array lengths before allocation. Check for circular references.",
        severity="error",
        tags="range,stack,recursion,array,javascript",
    ),

    # ── TypeScript ──────────────────────────────────────────────────────
    ErrorFix(
        error_type="TS2322",
        language="typescript",
        pattern=r"TS2322:\s*Type\s+'.*?'\s+is\s+not\s+assignable\s+to\s+type",
        cause="Type mismatch: the assigned value doesn't match the expected type.",
        fix_template="Check the expected type and cast or transform the value. "
                     "Use a type guard (if (x instanceof Y)) or assertion (x as Type) if safe.",
        severity="error",
        tags="type,assignable,mismatch,typescript",
    ),
    ErrorFix(
        error_type="TS2339",
        language="typescript",
        pattern=r"TS2339:\s*Property\s+'.*?'\s+does\s+not\s+exist\s+on\s+type",
        cause="Accessing a property not defined in the type declaration.",
        fix_template="Add the property to the type/interface definition. "
                     "Use optional chaining (obj?.prop) or extend the interface.",
        severity="error",
        tags="property,type,interface,missing,typescript",
    ),
    ErrorFix(
        error_type="TS2345",
        language="typescript",
        pattern=r"TS2345:\s*Argument\s+of\s+type\s+'.*?'\s+is\s+not\s+assignable\s+to\s+parameter",
        cause="Function argument type doesn't match parameter type.",
        fix_template="Transform the argument to match the expected type. "
                     "Use generics or overloads if multiple types are valid.",
        severity="error",
        tags="argument,parameter,type,function,typescript",
    ),
    ErrorFix(
        error_type="TS7006",
        language="typescript",
        pattern=r"TS7006:\s*Parameter\s+'.*?'\s+implicitly\s+has\s+an\s+'any'\s+type",
        cause="noImplicitAny is enabled and the parameter lacks a type annotation.",
        fix_template="Add explicit type annotations to function parameters: "
                     "function foo(param: string) instead of function foo(param).",
        severity="warning",
        tags="any,implicit,annotation,noImplicitAny,typescript",
    ),
    ErrorFix(
        error_type="TS2304",
        language="typescript",
        pattern=r"TS2304:\s*Cannot\s+find\s+name\s+'.*?'",
        cause="Identifier not found — missing import, declaration, or type definition.",
        fix_template="Import the missing symbol or install its @types/ package. "
                     "Check tsconfig.json 'lib' and 'typeRoots' settings.",
        severity="error",
        tags="name,import,declaration,types,typescript",
    ),

    # ── Java ────────────────────────────────────────────────────────────
    ErrorFix(
        error_type="NullPointerException",
        language="java",
        pattern=r"(NullPointerException|java\.lang\.NullPointerException)",
        cause="Dereferencing a null reference.",
        fix_template="Add null checks before method calls: if (obj != null). "
                     "Use Optional<T> for values that may be absent. "
                     "Enable @Nullable/@NonNull annotations.",
        severity="error",
        tags="null,pointer,npe,reference,java",
    ),
    ErrorFix(
        error_type="ClassCastException",
        language="java",
        pattern=r"ClassCastException:\s*.*cannot\s+be\s+cast\s+to",
        cause="Invalid type cast between incompatible classes.",
        fix_template="Use instanceof check before casting: "
                     "if (obj instanceof MyClass) { MyClass m = (MyClass) obj; }. "
                     "Prefer generics over raw types.",
        severity="error",
        tags="cast,class,type,instanceof,java",
    ),
    ErrorFix(
        error_type="StackOverflowError",
        language="java",
        pattern=r"(StackOverflowError|java\.lang\.StackOverflowError)",
        cause="Recursive call without proper termination or extremely deep call stack.",
        fix_template="Add a proper base case to recursive methods. "
                     "Convert deep recursion to iteration. "
                     "Increase stack size with -Xss flag if legitimate.",
        severity="error",
        tags="stack,overflow,recursion,depth,java",
    ),
    ErrorFix(
        error_type="ArrayIndexOutOfBoundsException",
        language="java",
        pattern=r"ArrayIndexOutOfBoundsException:\s*Index\s+\d+\s+out\s+of\s+bounds",
        cause="Array index is negative or >= array.length.",
        fix_template="Check array bounds before access: "
                     "if (i >= 0 && i < arr.length). "
                     "Use enhanced for-loop (for-each) when possible.",
        severity="error",
        tags="array,index,bounds,outofbounds,java",
    ),
    ErrorFix(
        error_type="ConcurrentModificationException",
        language="java",
        pattern=r"ConcurrentModificationException",
        cause="Collection modified while being iterated.",
        fix_template="Use Iterator.remove() for removal during iteration. "
                     "Use ConcurrentHashMap or CopyOnWriteArrayList for concurrent access. "
                     "Collect items to remove and remove after iteration.",
        severity="error",
        tags="concurrent,modification,iterator,collection,java",
    ),

    # ── Go ──────────────────────────────────────────────────────────────
    ErrorFix(
        error_type="nil pointer dereference",
        language="go",
        pattern=r"(nil\s+pointer\s+dereference|invalid\s+memory\s+address\s+or\s+nil\s+pointer)",
        cause="Dereferencing a nil pointer.",
        fix_template="Always check for nil before dereferencing: "
                     "if ptr != nil { use ptr }. "
                     "Return (value, error) tuples and check errors.",
        severity="error",
        tags="nil,pointer,dereference,null,go",
    ),
    ErrorFix(
        error_type="index out of range",
        language="go",
        pattern=r"index\s+out\s+of\s+range\s*\[?\d*\]?",
        cause="Slice or array index exceeds length.",
        fix_template="Check slice length before access: "
                     "if i < len(slice) { use slice[i] }. "
                     "Use range loops to avoid manual indexing.",
        severity="error",
        tags="index,range,slice,array,bounds,go",
    ),
    ErrorFix(
        error_type="goroutine leak",
        language="go",
        pattern=r"(goroutine\s+leak|too\s+many\s+goroutines|all\s+goroutines\s+are\s+asleep)",
        cause="Goroutine blocked forever on channel operation or never terminates.",
        fix_template="Use context.WithCancel or context.WithTimeout for goroutine lifecycle. "
                     "Ensure channels are closed when no more values will be sent. "
                     "Use select with a done channel for graceful shutdown.",
        severity="error",
        tags="goroutine,leak,channel,deadlock,go",
    ),
    ErrorFix(
        error_type="data race",
        language="go",
        pattern=r"(DATA\s+RACE|data\s+race|race\s+detected)",
        cause="Concurrent unsynchronized access to shared memory.",
        fix_template="Protect shared state with sync.Mutex or sync.RWMutex. "
                     "Use channels for goroutine communication. "
                     "Run tests with -race flag: go test -race ./...",
        severity="error",
        tags="race,concurrent,mutex,sync,go",
    ),
    ErrorFix(
        error_type="deadlock",
        language="go",
        pattern=r"fatal\s+error:\s+all\s+goroutines\s+are\s+asleep\s*-\s*deadlock",
        cause="All goroutines are blocked — no goroutine can make progress.",
        fix_template="Check for circular channel dependencies. "
                     "Use buffered channels or select with default case. "
                     "Ensure WaitGroup.Done() is called for every Add().",
        severity="error",
        tags="deadlock,goroutine,channel,waitgroup,go",
    ),

    # ── Rust ────────────────────────────────────────────────────────────
    ErrorFix(
        error_type="E0382",
        language="rust",
        pattern=r"(E0382|borrow\s+of\s+moved\s+value|use\s+of\s+moved\s+value)",
        cause="Value used after being moved (ownership transferred).",
        fix_template="Clone the value with .clone() if needed in multiple places. "
                     "Use references (&T or &mut T) instead of transferring ownership. "
                     "Restructure code to avoid needing the value after the move.",
        severity="error",
        tags="borrow,move,ownership,value,rust",
    ),
    ErrorFix(
        error_type="E0502",
        language="rust",
        pattern=r"(E0502|cannot\s+borrow.*as\s+(im)?mutable.*also\s+borrowed)",
        cause="Conflicting borrows: cannot have mutable and immutable borrow simultaneously.",
        fix_template="Restructure to avoid overlapping borrows. "
                     "Use scoping to limit borrow lifetimes. "
                     "Consider using Cell<T> or RefCell<T> for interior mutability.",
        severity="error",
        tags="borrow,mutable,immutable,reference,rust",
    ),
    ErrorFix(
        error_type="E0308",
        language="rust",
        pattern=r"(E0308|mismatched\s+types|expected\s+.*,\s+found)",
        cause="Type mismatch between expected and actual types.",
        fix_template="Check the expected return type. Use .into() or From/Into traits "
                     "for conversions. Use as keyword for primitive casts.",
        severity="error",
        tags="type,mismatch,expected,found,rust",
    ),
    ErrorFix(
        error_type="E0599",
        language="rust",
        pattern=r"(E0599|no\s+method\s+named\s+.*\s+found\s+for)",
        cause="Method not found on the type — missing trait import or wrong type.",
        fix_template="Import the trait that provides the method with 'use TraitName;'. "
                     "Check the type implements the required trait. "
                     "Verify you're calling on the correct type (not a reference or wrapper).",
        severity="error",
        tags="method,trait,impl,found,rust",
    ),
    ErrorFix(
        error_type="thread panic",
        language="rust",
        pattern=r"(thread\s+'.*'\s+panicked|unwrap\(\)\s+on\s+a\s+`(None|Err)`)",
        cause="Panicked due to unwrap() on None or Err, or explicit panic!().",
        fix_template="Replace unwrap() with pattern matching or ? operator. "
                     "Use .unwrap_or_default() or .expect('message') for clearer errors. "
                     "Handle Result/Option types explicitly.",
        severity="error",
        tags="panic,unwrap,none,err,thread,rust",
    ),

    # ── C# ──────────────────────────────────────────────────────────────
    ErrorFix(
        error_type="NullReferenceException",
        language="csharp",
        pattern=r"(NullReferenceException|Object\s+reference\s+not\s+set\s+to\s+an\s+instance)",
        cause="Accessing a member on a null object reference.",
        fix_template="Use null-conditional operator: obj?.Method(). "
                     "Enable nullable reference types (#nullable enable). "
                     "Check for null before access or use ?? for defaults.",
        severity="error",
        tags="null,reference,object,instance,csharp",
    ),
    ErrorFix(
        error_type="InvalidCastException",
        language="csharp",
        pattern=r"InvalidCastException:\s*Unable\s+to\s+cast",
        cause="Invalid type cast between incompatible types.",
        fix_template="Use 'as' operator with null check: var x = obj as MyType; if (x != null). "
                     "Or use 'is' pattern: if (obj is MyType x) { use x }.",
        severity="error",
        tags="cast,type,invalid,csharp",
    ),
    ErrorFix(
        error_type="StackOverflowException",
        language="csharp",
        pattern=r"StackOverflowException",
        cause="Infinite recursion or very deep call stack.",
        fix_template="Add a base case to recursive methods. "
                     "Convert to iterative with explicit stack. "
                     "Check for property getter/setter calling itself.",
        severity="error",
        tags="stack,overflow,recursion,csharp",
    ),
    ErrorFix(
        error_type="ArgumentNullException",
        language="csharp",
        pattern=r"ArgumentNullException:\s*Value\s+cannot\s+be\s+null",
        cause="A null argument was passed to a method that doesn't accept null.",
        fix_template="Validate arguments with ArgumentNullException.ThrowIfNull(). "
                     "Add null checks at method entry. "
                     "Use [NotNull] attribute for compile-time checking.",
        severity="error",
        tags="argument,null,parameter,validation,csharp",
    ),
    ErrorFix(
        error_type="TaskCanceledException",
        language="csharp",
        pattern=r"(TaskCanceledException|OperationCanceledException)",
        cause="An async operation was canceled via CancellationToken or timed out.",
        fix_template="Handle TaskCanceledException in try/catch around async calls. "
                     "Check CancellationToken.IsCancellationRequested before long operations. "
                     "Set appropriate timeouts with CancellationTokenSource.",
        severity="error",
        tags="task,canceled,async,timeout,cancellation,csharp",
    ),
    # ── Tooling / Framework (all languages) ──────────────────────────────
    ErrorFix(
        error_type="TailwindCSSDeprecatedInit",
        language="all",
        pattern=r"(tailwindcss\s+init|npx\s+tailwindcss\s+init|tailwind\.config\.(js|ts|cjs|mjs))",
        cause="Tailwind CSS v4 removed the 'tailwindcss init' command and tailwind.config.js. "
              "The old v3 setup is no longer supported.",
        fix_template="Use the new Tailwind CSS v4 installation as a PostCSS plugin:\n"
                     "1. Install packages: npm install tailwindcss @tailwindcss/postcss postcss\n"
                     "2. Add Tailwind to your postcss.config.mjs:\n"
                     "   export default { plugins: { \"@tailwindcss/postcss\": {} } }\n"
                     "3. Add @import \"tailwindcss\"; to your main CSS file.\n"
                     "4. Configuration is now done directly in CSS using the @theme directive, NOT tailwind.config.js",
        severity="error",
        tags="tailwindcss,tailwind,init,config,postcss,css,deprecated,v4",
    ),
    ErrorFix(
        error_type="NpmUnknownCommand",
        language="javascript",
        pattern=r'Unknown command:?\s*"?(set-script|access|adduser|bin|birthday|bugs|cache|ci|completion|config|dedupe|deprecate|diff|dist-tag|docs|doctor|edit|exec|explain|explore|find-dupes|fund|get|help|hook|init|install|link|ll|login|logout|ls|org|outdated|owner|pack|ping|pkg|prefix|profile|prune|publish|query|rebuild|repo|restart|root|run|search|shrinkwrap|star|stars|start|stop|team|test|token|uninstall|unpublish|unstar|update|version|view|whoami)"?',
        cause="The npm subcommand is unknown — it may have been removed or renamed in a newer npm version.",
        fix_template="Common npm command replacements:\n"
                     "- `npm set-script <name> <cmd>` → removed in npm v7+, use: npm pkg set scripts.<name>=\"<cmd>\"\n"
                     "- `npm adduser` → `npm login`\n"
                     "- If no replacement exists, edit package.json directly to achieve the same effect.",
        severity="error",
        tags="npm,unknown,command,deprecated,set-script,pkg",
    ),
    ErrorFix(
        error_type="NpmSetScriptDeprecated",
        language="javascript",
        pattern=r'(npm\s+set-script|Unknown command:?\s*"?set-script"?)',
        cause="`npm set-script` was removed in npm v7+. It no longer exists as a subcommand.",
        fix_template="Replace `npm set-script <name> \"<command>\"` with:\n"
                     "  npm pkg set scripts.<name>=\"<command>\"\n\n"
                     "Examples:\n"
                     "  npm pkg set scripts.start=\"vite\"\n"
                     "  npm pkg set scripts.build=\"vite build\"\n"
                     "  npm pkg set scripts.dev=\"vite --open\"",
        severity="error",
        tags="npm,set-script,deprecated,pkg,scripts,package-json",
    ),
    ErrorFix(
        error_type="NpmMissingScript",
        language="javascript",
        pattern=r'npm error Missing script: "(\w+)"',
        cause="The requested script is not defined in the 'scripts' section of package.json.",
        fix_template="The script '{1}' is missing from package.json.\n\n"
                     "FIX OPTIONS:\n"
                     "1. IF TESTING WITH VITEST: Run 'npx vitest' or 'npx vitest run {1}' directly.\n"
                     "2. IF TESTING WITH JEST: Run 'npx jest' directly.\n"
                     "3. ADD THE SCRIPT: Edit package.json and add \"{1}\": \"<command>\" to the 'scripts' section.\n"
                     "   Example: npm pkg set scripts.{1}=\"vitest run\"\n\n"
                     "Check if you are in the correct directory (sub-project root) where package.json resides.",
        severity="error",
        tags="npm,script,missing,package-json,test,vitest,jest",
    ),
    ErrorFix(
        error_type="TailwindCSSDeprecatedDirectives",
        language="all",
        pattern=r"@tailwind\s+(base|components|utilities)",
        cause="The @tailwind base, @tailwind components, and @tailwind utilities directives "
              "are REMOVED in Tailwind CSS v4. They no longer exist and will cause build errors. "
              "This is the #1 most common Tailwind v4 migration mistake.",
        fix_template="REPLACE the entire CSS file content with ONLY this single line:\n"
                     "@import \"tailwindcss\";\n\n"
                     "WRONG (v3 — do NOT use):\n"
                     "@tailwind base;\n"
                     "@tailwind components;\n"
                     "@tailwind utilities;\n\n"
                     "CORRECT (v4):\n"
                     "@import \"tailwindcss\";\n\n"
                     "Do NOT add @tailwind directives alongside @import \"tailwindcss\". "
                     "The single @import line replaces ALL three @tailwind directives. "
                     "Any custom CSS (e.g. body styles, @layer) should go AFTER the @import line.",
        severity="error",
        tags="tailwindcss,tailwind,css,directive,base,components,utilities,deprecated,v4,import",
    ),
    ErrorFix(
        error_type="AngularFlexLayoutDeprecated",
        language="all",
        pattern=r"No matching version found for @angular/flex-layout",
        cause="@angular/flex-layout is deprecated and does not support modern Angular versions (15+).",
        fix_template="Remove @angular/flex-layout from your npm install command and package.json. "
                     "For responsive layouts in modern Angular, use standard CSS Flexbox, CSS Grid, or Tailwind CSS instead.",
        severity="error",
        tags="angular,flex-layout,deprecated,npm,install,flexbox,grid",
    ),
    # ── React Testing Library — common test-source mismatch patterns ────
    ErrorFix(
        error_type="TestingLibraryElementNotFound",
        language="javascript",
        pattern=r"TestingLibraryElementError:\s*Unable to find an (accessible )?element",
        cause="The test asserts text, role, or label that does not exist in the component's "
              "actual rendered output. Common causes: (1) the test hardcodes text content the "
              "component doesn't render, (2) text is split across multiple DOM elements so "
              "exact string matching fails, (3) the element uses a different role/tag than expected, "
              "(4) the rendered DOM is EMPTY or only contains <body> </div> </body>.",
        fix_template="CRITICAL FIX STEPS:\n"
                     "1. READ THE ACTUAL DOM OUTPUT shown in the error message — look specifically "
                     "at the <body> section. If it only shows '<div> </div>' or '<body> </body>', "
                     "refer to TestingLibraryEmptyRootError.\n"
                     "2. Ignore the 'text is broken up by multiple elements' generic hint unless "
                     "you verified the text is actually split in the source.\n"
                     "3. Match your query to ACTUAL rendered text, not assumed text.\n"
                     "4. For text split across elements, use a function matcher:\n"
                     "   screen.getByText((content, element) => element?.textContent === 'expected full text')\n"
                     "5. Prefer getByRole() over getByText() for more robust queries:\n"
                     "   screen.getByRole('button', { name: /submit/i })\n"
                     "   screen.getByRole('heading', { name: /welcome/i })\n"
                     "6. NEVER hardcode assumed text content — always derive expected values "
                     "from the actual source component props or rendered DOM.\n"
                     "7. Use { exact: false } for partial text matching: screen.getByText('partial', { exact: false })",
        severity="error",
        tags="testing-library,react,getByText,getByRole,element,not,found,query,dom,javascript,typescript,empty-dom",
    ),
    ErrorFix(
        error_type="TestingLibraryRoleNameNotFound",
        language="javascript",
        pattern=r"TestingLibraryElementError:\s*Unable to find role=.*?(and name)?",
        cause="The test asserts an element with a specific role and name that does not exist in the "
              "component's rendered output. This is often due to the text changing, the element using a "
              "different role, or the component rendering an empty state.",
        fix_template="CRITICAL FIX STEPS:\n"
                     "1. READ THE ACTUAL DOM OUTPUT in the error message to see what was actually rendered.\n"
                     "2. Match your query to ACTUAL rendered text and roles, not assumed ones.\n"
                     "3. Check if the element uses a different role (e.g. 'link' instead of 'heading') "
                     "or if the text is split across multiple elements.\n"
                     "4. If text is split, use a function matcher: screen.getByText((content, element) => ...)\n"
                     "5. NEVER hardcode assumed text content — derive expected values from the component.",
        severity="error",
        tags="testing-library,react,getByRole,role,name,not,found,query,dom,javascript,typescript",
    ),
    ErrorFix(
        error_type="TestingLibraryEmptyRootError",
        language="javascript",
        pattern=r"<body>\s*(<\/div>)?\s*<\/body>",
        cause="The rendered DOM is empty or contains only the root <body> tag. This means "
              "the component failed to render any UI. Common causes: (1) missing required "
              "Context Providers (Router, Theme, Redux) causing the component to return null, "
              "(2) synchronous queries on a lazy-loaded component (React.lazy), "
              "(3) a bug in the component's conditional rendering (returning null/undefined), "
              "(4) incorrect import/export causing an 'undefined' component to be rendered.",
        fix_template="CRITICAL FIX STEPS:\n"
                     "1. CHECK FOR CONTEXT: Does the component use useNavigate, useLocation, or "
                     "custom hooks? Wrap it in <MemoryRouter> or the required <Provider>.\n"
                     "2. CHECK FOR LAZY LOADING: Is the component lazy-loaded? Use findBy* "
                     "(async) queries instead of getBy* (sync).\n"
                     "3. CHECK COMPONENT SOURCE: Does the component return null under certain "
                     "prop/state conditions? Verify you are passing the correct props in render().\n"
                     "4. CHECK IMPORTS: Ensure you didn't mix up default and named imports. "
                     "Rendering <undefined /> results in an empty DOM.\n"
                     "5. USE screen.debug(): In development, use screen.debug() to see why the "
                     "DOM is empty during the test run.",
        severity="error",
        tags="testing-library,react,empty,dom,body,root,null,render,javascript,typescript,lazy-load,context",
    ),
    ErrorFix(
        error_type="TailwindClassRegexBrittle",
        language="javascript",
        pattern=r"AssertionError:.*(toMatch|to\s+match).*(/\\b-|md\\\\:)",
        cause="The test uses brittle regex patterns to assert CSS classes. "
              "Common failures: (1) `\\b-translate` fails because `\\b` does not match "
              "the boundary before a hyphen in many environments. (2) Tailwind responsive "
              "classes (md:) require complex escaping in regex (e.g. `md\\\\:flex`). "
              "Asserting on `className` directly is highly discouraged.",
        fix_template="FIX: Use @testing-library/jest-dom's `toHaveClass()` instead of regex.\n\n"
                     "WRONG (brittle regex):\n"
                     "  expect(nav.className).toMatch(/\\b-translate-x-full\\b/)\n"
                     "  expect(nav.className).toMatch(/md\\\\:flex/)\n\n"
                     "CORRECT (robust matcher):\n"
                     "  expect(nav).toHaveClass('-translate-x-full')\n"
                     "  expect(nav).toHaveClass('md:flex')\n\n"
                     "The `toHaveClass` matcher correctly handles class list parsing and "
                     "doesn't require complex escaping or word-boundary logic.",
        severity="error",
        tags="tailwind,css,className,regex,toHaveClass,brittle,assertion,javascript,typescript",
    ),
    # ── toHaveClass asserts classes not present in source component ──────
    ErrorFix(
        error_type="ToHaveClassMismatch",
        language="javascript",
        pattern=r"Expected the element to have class:",
        cause="The test asserts CSS classes (e.g. rounded-2xl, shadow-lg, p-8) that do NOT exist "
              "on the target element in the source component. The test was generated by assuming "
              "what styling the component would have, rather than reading the actual source. "
              "Common cause: the LLM generated test assertions based on expected design aesthetics "
              "instead of the real className values in the JSX/TSX source.",
        fix_template="CRITICAL — THIS IS A TEST BUG, NOT A SOURCE BUG.\n"
                     "The test asserts CSS classes that the source element does not have.\n\n"
                     "FIX STEPS:\n"
                     "1. READ the actual source component file to find the element being queried "
                     "(e.g. the element matched by querySelector('.max-w-md')).\n"
                     "2. Check the REAL className/class attribute on that element in the JSX.\n"
                     "3. Update the test to assert ONLY classes that actually exist on the element. "
                     "Example: if the element has 'w-full max-w-md', assert those — not rounded-2xl.\n"
                     "4. If the element has no specific styling worth asserting, REMOVE the class "
                     "assertion entirely or replace with: expect(card).toBeTruthy()\n"
                     "5. NEVER add CSS classes to the source component just to satisfy a test.\n"
                     "6. Do NOT change the test selector unless the selector itself is wrong.\n\n"
                     "EXAMPLE:\n"
                     "  // WRONG: asserts classes that aren't on the element\n"
                     "  expect(card).toHaveClass('rounded-2xl')  // ← REMOVE\n"
                     "  expect(card).toHaveClass('shadow-lg')    // ← REMOVE\n\n"
                     "  // CORRECT: assert actual classes from source, or just existence\n"
                     "  expect(card).toHaveClass('max-w-md')     // ← real class\n"
                     "  // OR remove the assertion entirely if styling isn't relevant",
        severity="error",
        tags="toHaveClass,css,class,mismatch,assertion,test-bug,tailwind,styling,react,javascript,typescript",
    ),
    ErrorFix(
        error_type="TestingLibraryMultipleElementsFound",
        language="javascript",
        pattern=r"TestingLibraryElementError:\s*Found multiple elements with the (role|text)",
        cause="The test uses getByRole/getByText/getByText(fn) which expects exactly ONE matching "
              "element, but multiple elements match. Common causes: (1) a navigation bar and page "
              "content both have a link with the same name, (2) header/footer duplicate links, "
              "(3) nested components render the same elements, (4) using a getByText() function "
              "matcher with element.textContent.includes() — parent elements also match because "
              "textContent includes ALL descendant text, so every ancestor of the target element "
              "matches the predicate too.",
        fix_template="FIX STEPS:\n"
                     "1. **Function matcher matching parents (most common with getByText)**: "
                     "When using getByText((content, element) => element?.textContent?.includes('...')) "
                     "every ancestor element also matches because textContent includes all descendant "
                     "text. FIX: use getAllByText() and pick the innermost (last/first) match, OR "
                     "add a tag filter to select only leaf/specific elements:\n"
                     "   screen.getByText((content, element) => {\n"
                     "     return element?.tagName === 'SPAN' && element?.textContent?.includes('Jane Doe')\n"
                     "   })\n"
                     "   // OR use getAllByText and take first:\n"
                     "   const matches = screen.getAllByText((content, element) => {\n"
                     "     return element?.textContent?.includes('Jane Doe')\n"
                     "   })\n"
                     "   expect(matches[0]).toBeInTheDocument()\n"
                     "2. Use getAllByRole()/getAllByText() when multiple matches are expected, "
                     "then assert on the array:\n"
                     "   const links = screen.getAllByRole('link', { name: /home/i })\n"
                     "   expect(links).toHaveLength(2)\n"
                     "   expect(links[0]).toBeInTheDocument()\n"
                     "3. Scope the query to a specific container using within():\n"
                     "   import { within } from '@testing-library/react'\n"
                     "   const nav = screen.getByRole('navigation')\n"
                     "   within(nav).getByRole('link', { name: /home/i })\n"
                     "4. Use a more specific name pattern to narrow matches:\n"
                     "   screen.getByRole('link', { name: /^home$/i })  // exact match\n"
                     "5. READ the actual component source to understand how many instances exist "
                     "and which container to scope your query to.",
        severity="error",
        tags="testing-library,react,multiple,elements,getByRole,getAllByRole,getByText,getAllByText,textContent,function,matcher,within,query,javascript,typescript",
    ),
    # ── Assumed ARIA roles / labels that don't exist in the source ─────
    ErrorFix(
        error_type="TestingLibraryAssumedAriaNotFound",
        language="javascript",
        pattern=r"TestingLibraryElementError:\s*Unable to find (a label|an accessible element|a[n]? element) with the (text|role|label)",
        cause="The test assumes the component uses ARIA roles or labels (aria-label, role='region', "
              "aria-labelledby) that do NOT exist in the actual source code. This is the #1 cause of "
              "test failures in component testing: the LLM invents ARIA attributes that were never added "
              "to the component. Common cases:\n"
              "  (1) <section> without aria-label does NOT expose the 'region' role — "
              "getByRole('region', { name: /stats/i }) will fail.\n"
              "  (2) <div> has no implicit ARIA role — getByRole('region') won't find it.\n"
              "  (3) getByLabelText(/stats/i) fails because no element has aria-label='stats'.\n"
              "  (4) Plain HTML elements (div, section, span) rarely have ARIA attributes "
              "unless the developer explicitly added them.",
        fix_template="CRITICAL FIX STEPS:\n"
                     "1. READ the actual component source code — check which HTML elements are used "
                     "and whether they have aria-label, role, or aria-labelledby attributes.\n"
                     "2. NEVER assume ARIA attributes exist. Most components use plain HTML without "
                     "explicit ARIA roles or labels.\n"
                     "3. HTML implicit roles ONLY apply when an accessible name is present:\n"
                     "   - <section aria-label='stats'> → role='region', findable by getByRole('region', { name: /stats/i })\n"
                     "   - <section> (no aria-label) → NO implicit role, NOT findable by getByRole('region')\n"
                     "   - <nav> → always role='navigation' (no label required)\n"
                     "   - <main> → always role='main'\n"
                     "   - <header> → role='banner' (when not nested in sectioning content)\n"
                     "   - <footer> → role='contentinfo' (when not nested in sectioning content)\n"
                     "   - <div> → NO implicit role ever\n"
                     "4. Instead of querying by assumed ARIA role, use alternatives:\n"
                     "   - Query by actual text content: screen.getByText(/total users/i)\n"
                     "   - Query by heading: screen.getByRole('heading', { name: /stats/i })\n"
                     "   - Query by test-id if present: screen.getByTestId('stats-section')\n"
                     "   - Query the container structure: check what's inside <main> or look for \n"
                     "     specific child content that proves the section rendered.\n"
                     "5. If the component truly renders a <section> without aria-label, do NOT use \n"
                     "   getByRole('region'). Instead, verify the section's CONTENT exists:\n"
                     "   // Instead of: screen.getByRole('region', { name: /stats/i })\n"
                     "   // Do: verify actual stats content is rendered\n"
                     "   expect(screen.getByText(/total users/i)).toBeInTheDocument()\n"
                     "   expect(screen.getByText(/revenue/i)).toBeInTheDocument()",
        severity="error",
        tags="testing-library,react,aria,role,region,label,section,getByRole,getByLabelText,accessible,name,assumed,not,found,javascript,typescript",
    ),
    ErrorFix(
        error_type="ReactRouterContextMissing",
        language="javascript",
        pattern=r"(useLocation|useNavigate|useParams|useMatch|useHref)\(\) may be used only in the context of a <Router> component",
        cause="The component uses React Router hooks (useLocation, useNavigate, useParams, etc.) "
              "or components (Link, NavLink, Outlet) but the component is not wrapped in a Router provider. "
              "React Router hooks MUST be called inside a Router context.",
        fix_template="FIX DEPENDS ON CONTEXT:\n\n"
                     "1. IN APPLICATION CODE (main.jsx or App.jsx):\n"
                     "Ensure your entire app is wrapped in a <BrowserRouter> at the highest level (typically main.jsx/index.jsx):\n"
                     "import { BrowserRouter } from 'react-router-dom';\n"
                     "ReactDOM.createRoot(document.getElementById('root')).render(\n"
                     "  <BrowserRouter><App /></BrowserRouter>\n"
                     ");\n\n"
                     "2. IN TESTS (.test.jsx or .spec.jsx):\n"
                     "Wrap the tested component in <MemoryRouter> (do not use BrowserRouter in tests):\n"
                     "import { MemoryRouter } from 'react-router-dom';\n"
                     "render(<MemoryRouter initialEntries={['/']}><YourComponent /></MemoryRouter>);\n\n"
                     "KEY RULES:\n"
                     "- ALWAYS use MemoryRouter in tests.\n"
                     "- Ensure <BrowserRouter> is only used ONCE in the runtime app tree.\n"
                     "- Any component calling useLocation(), useNavigate(), or containing <Routes>/<Route> MUST be a child of a Router.",
        severity="error",
        tags="react-router,useLocation,useNavigate,useParams,MemoryRouter,BrowserRouter,Router,context,testing,react,javascript,typescript",
    ),
    # ── Nested Router: test wraps App that already has its own Router ─────
    ErrorFix(
        error_type="ReactRouterNestedRouter",
        language="javascript",
        pattern=r"You cannot render a <Router> inside another <Router>",
        cause="The test wraps the component in <MemoryRouter> but the component (usually App) "
              "already contains its own <BrowserRouter> or <HashRouter>. React Router does not "
              "allow nested Router providers. This typically happens when testing an App component "
              "that defines its own Router internally — the test adds a second one around it.",
        fix_template="CRITICAL FIX STEPS:\n"
                     "1. READ the App/component source to check if it already includes a Router "
                     "(BrowserRouter, HashRouter, or Router).\n"
                     "2. If the App ALREADY has a Router:\n"
                     "   Option A — Render App directly WITHOUT MemoryRouter wrapper:\n"
                     "     render(<App />)\n"
                     "     // But this uses BrowserRouter which is hard to test.\n\n"
                     "   Option B (PREFERRED) — Mock react-router-dom to replace BrowserRouter "
                     "with MemoryRouter so the test controls routing:\n"
                     "     vi.mock('react-router-dom', async () => {\n"
                     "       const actual = await vi.importActual('react-router-dom')\n"
                     "       return { ...actual, BrowserRouter: ({ children }) => (\n"
                     "         <actual.MemoryRouter initialEntries={['/']}>{children}</actual.MemoryRouter>\n"
                     "       )}\n"
                     "     })\n"
                     "     render(<App />)  // No extra MemoryRouter wrapper needed\n\n"
                     "   Option C — Test individual page components instead of App:\n"
                     "     render(\n"
                     "       <MemoryRouter initialEntries={['/']}>\n"
                     "         <Homepage />\n"
                     "       </MemoryRouter>\n"
                     "     )\n\n"
                     "3. If the App does NOT have a Router (just exports Routes), "
                     "then wrapping in MemoryRouter is correct — the error is elsewhere.\n"
                     "4. NEVER nest two Routers. One Router per render tree.",
        severity="error",
        tags="react-router,BrowserRouter,MemoryRouter,nested,Router,inside,another,testing,react,App,javascript,typescript",
    ),
    ErrorFix(
        error_type="TestAssertionLengthMismatch",
        language="javascript",
        pattern=r"expected .* to have a length of \d+ but got \d+",
        cause="The test hardcodes an expected element count (e.g. toHaveLength(3)) but the "
              "component renders a different number of elements. The LLM assumed a count "
              "without reading the actual source code.",
        fix_template="FIX STEPS:\n"
                     "1. Do NOT hardcode expected element counts in tests.\n"
                     "2. Read the ACTUAL component source to determine how many elements it renders.\n"
                     "3. If elements come from props/data, check the test's mock data length.\n"
                     "4. Use more specific selectors to narrow down:\n"
                     "   - screen.getAllByRole('button', { name: /buy/i }) instead of screen.getAllByRole('button')\n"
                     "   - within(container).getAllByRole('listitem') to scope to a section\n"
                     "5. If the count is dynamic, assert against the test data length:\n"
                     "   expect(items).toHaveLength(mockData.length)\n"
                     "6. Check for duplicate elements from nested components.",
        severity="error",
        tags="assertion,length,count,toHaveLength,mismatch,testing,react,javascript,typescript",
    ),
    ErrorFix(
        error_type="MockCallbackNotCalled",
        language="javascript",
        pattern=r'expected "(vi\.fn|jest\.fn)\(\)" to (be called|have been called)',
        cause="The test expects a mock callback (vi.fn()/jest.fn()) to be called when an "
              "element is clicked, but the component does not call it. Common causes: "
              "(1) the prop name in the test doesn't match the component's prop name, "
              "(2) the component doesn't wire the callback to an onClick handler, "
              "(3) the event target element isn't the one with the handler (event delegation), "
              "(4) the component uses a <Link to='...'> instead of useNavigate() so the "
              "mocked navigate function is never called.",
        fix_template="FIX STEPS:\n"
                     "1. READ the actual component source to verify the EXACT prop names for callbacks.\n"
                     "   Common mismatches: onCtaClick vs onClick, handleSubmit vs onSubmit\n"
                     "2. Verify the component actually passes the callback to an onClick/onChange handler.\n"
                     "3. Use userEvent (not fireEvent) for realistic interaction:\n"
                     "   await userEvent.click(button)\n"
                     "4. Check if the button is disabled — disabled buttons don't fire click events.\n"
                     "5. Check if the handler is conditional (e.g. only fires after form validation).\n"
                     "6. For event delegation, click the actual target element, not a parent container.\n"
                     "7. If the component uses a link (<a>) instead of <button>, the callback may be "
                     "on navigation, not onClick.\n"
                     "8. If mocking useNavigate but the component uses <Link to='...'>, the mock won't "
                     "be called — <Link> uses internal router navigation, not the useNavigate hook. "
                     "Instead, assert the link's href: expect(screen.getByRole('link')).toHaveAttribute('href', '/path')",
        severity="error",
        tags="mock,callback,vi.fn,jest.fn,click,called,event,handler,react,javascript,typescript",
    ),
    ErrorFix(
        error_type="ReactTestPropsNotMatching",
        language="javascript",
        pattern=r"(received.*undefined|expected.*undefined|Cannot read properties of undefined.*reading)",
        cause="The test passes props that don't match the component's expected prop interface, "
              "or accesses props/state that the component doesn't expose. The LLM assumed "
              "a component API without reading the actual source.",
        fix_template="FIX STEPS:\n"
                     "1. Read the component source to find the EXACT prop names and types.\n"
                     "2. Check if the component destructures props — misspelled prop names become undefined.\n"
                     "3. Ensure required props are provided in the test render:\n"
                     "   render(<Component requiredProp='value' />)\n"
                     "4. For components using React Router, wrap with <MemoryRouter>.\n"
                     "5. For components using context providers, wrap with the appropriate provider.\n"
                     "6. Check default prop values — the component may handle missing props gracefully.",
        severity="error",
        tags="props,undefined,component,react,render,testing,javascript,typescript",
    ),
    ErrorFix(
        error_type="ViteFailedToResolveImport",
        language="javascript",
        pattern=r"Failed to resolve import\s+\"[^\"]+\"",
        cause="Vite/Vitest cannot resolve an npm package import because the package "
              "is not installed in node_modules. This commonly happens with "
              "@testing-library/user-event, @testing-library/jest-dom, or other "
              "test utility packages that the LLM imports but weren't installed.",
        fix_template="FIX: Install the missing package as a dev dependency:\n"
                     "  npm install --save-dev <package-name>\n\n"
                     "Common missing packages for Vitest + React + Testing Library:\n"
                     "  npm install --save-dev vitest jsdom\n"
                     "  npm install --save-dev @testing-library/react @testing-library/dom\n"
                     "  npm install --save-dev @testing-library/jest-dom @testing-library/user-event\n"
                     "  npm install --save-dev @vitejs/plugin-react\n"
                     "  npm install --save-dev react-router-dom\n\n"
                     "CRITICAL packages often missed:\n"
                     "  - jsdom: Required for Vitest DOM environment (environment: 'jsdom' in config)\n"
                     "  - @testing-library/dom: Peer dep of @testing-library/react\n"
                     "  - @vitejs/plugin-react: Required for JSX transform in Vite/Vitest\n\n"
                     "Do NOT try to fix the import path or remove the import — "
                     "the package genuinely needs to be installed.",
        severity="error",
        tags="vite,vitest,import,resolve,module,not,found,install,npm,javascript,typescript",
    ),
    ErrorFix(
        error_type="NoTestSuiteFound",
        language="javascript",
        pattern=r"No test suite found in file",
        cause="The test file exists but contains no describe/it/test blocks. "
              "It is likely a setup file, configuration scaffold, or vitest/jest "
              "setup file (e.g. vitestSetup.test.js) that was mistakenly given "
              "a .test. extension. Test runners require at least one test block.",
        fix_template="FIX: Do NOT generate setup/scaffold files with .test. extensions.\n"
                     "1. DELETE or rename the file to remove the .test. part "
                     "(e.g. vitestSetup.test.js → vitest.setup.js).\n"
                     "2. Setup files should NOT have .test. in the name — use "
                     ".setup.js, .config.js, or conftest.py instead.\n"
                     "3. Every .test. or .spec. file MUST contain at least one "
                     "describe/it/test block with actual assertions.\n"
                     "4. If the file was meant to configure vitest, put it in "
                     "vitest.config.ts or a setup file referenced by setupFiles.",
        severity="error",
        tags="vitest,jest,test,suite,empty,setup,scaffold,no,found,javascript,typescript",
    ),
    # ── Vitest + @testing-library/jest-dom ──────────────────────────────
    ErrorFix(
        error_type="VitestJestDomExpectNotDefined",
        language="javascript",
        pattern=r"ReferenceError:\s*expect\s+is\s+not\s+defined.*@testing-library/jest-dom",
        cause="@testing-library/jest-dom calls expect.extend() at import time, but Vitest "
              "does not expose `expect` as a global unless `globals: true` is set in vitest.config. "
              "Importing '@testing-library/jest-dom' directly crashes in Vitest projects.",
        fix_template="Replace the import in ALL test files:\n\n"
                     "WRONG (crashes in Vitest):\n"
                     "  import '@testing-library/jest-dom';\n\n"
                     "CORRECT (Vitest-compatible, jest-dom v6+):\n"
                     "  import '@testing-library/jest-dom/vitest';\n\n"
                     "This import auto-registers jest-dom matchers (toBeInTheDocument, toHaveClass, etc.) "
                     "with Vitest's expect without requiring globals: true.\n\n"
                     "If using jest-dom v5 or earlier, upgrade to v6+:\n"
                     "  npm install @testing-library/jest-dom@latest\n\n"
                     "Alternatively, add a Vitest setup file:\n"
                     "  // vitest.setup.ts\n"
                     "  import '@testing-library/jest-dom/vitest';\n"
                     "  // vitest.config.ts: setupFiles: ['./vitest.setup.ts']",
        severity="error",
        tags="vitest,jest-dom,expect,testing-library,react,undefined,globals,javascript,typescript",
    ),
    ErrorFix(
        error_type="VitestExpectNotDefined",
        language="javascript",
        pattern=r"ReferenceError:\s*expect\s+is\s+not\s+defined",
        cause="Vitest does not expose test globals (expect, describe, it, etc.) by default. "
              "They must be explicitly imported from 'vitest', or globals: true must be set "
              "in vitest.config. This commonly happens when @testing-library/jest-dom is "
              "imported before vitest's expect is available.",
        fix_template="Fix test files in ONE of these ways:\n\n"
                     "Option 1 — Explicit imports (recommended, no config change):\n"
                     "  import { describe, it, expect, beforeEach, vi } from 'vitest';\n"
                     "  import '@testing-library/jest-dom/vitest';  // NOT '@testing-library/jest-dom'\n\n"
                     "Option 2 — Enable globals in vitest.config:\n"
                     "  // vitest.config.ts\n"
                     "  export default defineConfig({ test: { globals: true } });\n"
                     "  // Then add to tsconfig.json: \"types\": [\"vitest/globals\"]\n\n"
                     "IMPORTANT: If using @testing-library/jest-dom, you MUST use the "
                     "'/vitest' subpath import, not the bare import.",
        severity="error",
        tags="vitest,expect,globals,undefined,testing,ReferenceError,javascript,typescript",
    ),
    # ── queryByRole/queryByText returns null → toBeInTheDocument fails ──
    ErrorFix(
        error_type="QueryByRoleReturnedNull",
        language="javascript",
        pattern=r"(received value must be an HTMLElement or an SVGElement|Received has (type|value):\s*[Nn]ull)",
        cause="The test uses queryByRole/queryByText which returns null when no match is found, "
              "then asserts .toBeInTheDocument() on the null value. This means the queried element "
              "does NOT exist in the rendered DOM. The most common cause is the test assuming an "
              "element's role or accessible name based on the route path or page name rather than "
              "reading the actual component source. For example: a page at route '/dashboard' may "
              "NOT have a link with text 'Dashboard' — the heading might say 'Dashboard' (role=heading, "
              "not role=link), or sidebar links may use different labels like 'Home', 'Stats', etc.",
        fix_template="CRITICAL FIX STEPS:\n"
                     "1. READ the actual component source for the route being tested. The DOM output "
                     "in the error shows exactly what is rendered.\n"
                     "2. Do NOT assume element roles from route names:\n"
                     "   - A page at '/dashboard' does NOT necessarily have a <a>Dashboard</a> link.\n"
                     "   - The text 'Dashboard' might be in an <h1> (role=heading), not a link.\n"
                     "   - Navigation links may use labels different from the route path.\n"
                     "3. Use the CORRECT role for the actual element:\n"
                     "   - <h1>Dashboard</h1> → getByRole('heading', { name: /dashboard/i })\n"
                     "   - <a href='/dashboard'>Go to Dashboard</a> → getByRole('link', { name: /dashboard/i })\n"
                     "   - <button>Dashboard</button> → getByRole('button', { name: /dashboard/i })\n"
                     "4. If the element genuinely doesn't exist on this page/route, remove the assertion "
                     "or replace it with an assertion for an element that DOES exist.\n"
                     "5. Check if the element is in a DIFFERENT component that only renders on another route "
                     "(e.g., a Header with nav links only rendered by HomePage, not Dashboard).\n"
                     "6. Use screen.debug() or read the error's 'prettyDOM' output to see the actual DOM.",
        severity="error",
        tags="queryByRole,queryByText,null,toBeInTheDocument,role,heading,link,route,testing-library,react,javascript,typescript",
    ),
    # ── rerender() with new MemoryRouter does NOT navigate ─────────────
    ErrorFix(
        error_type="MemoryRouterRerenderNoNavigation",
        language="javascript",
        pattern=r"(route.*rerender|rerender.*route|rerender.*MemoryRouter|MemoryRouter.*rerender)",
        cause="Using rerender() with a NEW MemoryRouter does NOT navigate to a different route. "
              "React reconciliation keeps the old component tree because the Router is a new instance "
              "but React does not unmount/remount the children. The DOM still shows the PREVIOUS route's "
              "content. This is the #1 most common React Router testing mistake when LLMs generate tests.",
        fix_template="FIX: Do NOT use rerender() with a new MemoryRouter to test route changes.\n\n"
                     "WRONG (does not navigate — old route content persists):\n"
                     "  const { rerender } = render(\n"
                     "    <MemoryRouter initialEntries={['/']}><App /></MemoryRouter>\n"
                     "  )\n"
                     "  rerender(\n"
                     "    <MemoryRouter initialEntries={['/dashboard']}><App /></MemoryRouter>\n"
                     "  )\n"
                     "  // BUG: DOM still shows '/' content, NOT '/dashboard'\n\n"
                     "CORRECT — Option 1: Separate render calls (simplest):\n"
                     "  const { unmount } = render(\n"
                     "    <MemoryRouter initialEntries={['/']}><App /></MemoryRouter>\n"
                     "  )\n"
                     "  // Assert home page content...\n"
                     "  unmount()\n"
                     "  render(\n"
                     "    <MemoryRouter initialEntries={['/dashboard']}><App /></MemoryRouter>\n"
                     "  )\n"
                     "  // Now assert dashboard content — this is a fresh render\n\n"
                     "CORRECT — Option 2: Click a navigation link within the same router:\n"
                     "  render(\n"
                     "    <MemoryRouter initialEntries={['/']}><App /></MemoryRouter>\n"
                     "  )\n"
                     "  await userEvent.click(screen.getByRole('link', { name: /dashboard/i }))\n"
                     "  // Route changed within the same router — DOM updates correctly\n\n"
                     "KEY RULE: Each MemoryRouter is a separate routing context. rerender() swaps the "
                     "React tree but the new MemoryRouter's internal state does not carry over. "
                     "Always use unmount()+render() or navigate within the same router instance.",
        severity="error",
        tags="rerender,MemoryRouter,route,navigate,change,initialEntries,testing-library,react,react-router,javascript,typescript",
    ),
    # ── waitFor timeout — test assumes content the component never renders ──
    ErrorFix(
        error_type="WaitForAssumedContentNotRendered",
        language="javascript",
        pattern=r"Unable to find an element with the text:.*\(content,\s*element\)\s*=>",
        cause="The test uses waitFor() with a custom function matcher that asserts specific "
              "text content (e.g. 'No records found.', 'Loading...') that the component never "
              "actually renders. The LLM assumed what the component would display after async "
              "loading without reading the source. The waitFor() call times out because the "
              "expected DOM element never appears. This is a TEST_BUG, not a source bug.",
        fix_template="CRITICAL — THIS IS A TEST BUG, NOT A SOURCE BUG.\n"
                     "The test assumes the component renders text that it does not.\n\n"
                     "FIX STEPS:\n"
                     "1. READ the actual source component file to see what it REALLY renders "
                     "after loading completes (look for return statements, JSX output, "
                     "conditional rendering, empty states, loading states).\n"
                     "2. READ the DOM output shown in the error — it dumps the actual rendered "
                     "HTML. Find the real text/elements the component produces.\n"
                     "3. Rewrite the waitFor() assertion to match ACTUAL rendered content:\n"
                     "   // WRONG — assumed text:\n"
                     "   await waitFor(() => {\n"
                     "     screen.getByText((c, el) => el.tagName === 'P' && c === 'No records found.')\n"
                     "   })\n"
                     "   // CORRECT — use actual rendered text from source:\n"
                     "   await waitFor(() => {\n"
                     "     screen.getByText(/actual text from component/i)\n"
                     "   })\n"
                     "4. For loading → loaded transitions, verify the component's actual loading "
                     "indicator and post-load content by reading the source.\n"
                     "5. If the component renders a table or list, query by role instead:\n"
                     "   await waitFor(() => screen.getByRole('table'))\n"
                     "6. NEVER invent expected text content — always derive it from the "
                     "component source code.",
        severity="error",
        tags="waitFor,timeout,testing-library,react,async,loading,assumed,content,function,matcher,getByText,TEST_BUG,javascript,typescript",
    ),
    # ── Lazy-loaded component: sync query hits Suspense fallback ────────
    ErrorFix(
        error_type="SuspenseFallbackSyncQuery",
        language="javascript",
        pattern=r"Unable to find an accessible element with the role .*(Loading|Suspense|loading)",
        cause="The component under test is lazy-loaded with React.lazy() and wrapped in "
              "<Suspense fallback={...}>. The test uses a SYNCHRONOUS query (getByRole, getByText) "
              "which runs immediately and finds only the Suspense fallback (e.g. 'Loading dashboard...') "
              "instead of the actual component content. The lazy chunk has not finished loading yet "
              "when the sync query executes, so elements like headings, tables, navigation, etc. "
              "do not exist in the DOM at query time. This is common when App.jsx uses React.lazy() "
              "for route-level code splitting.",
        fix_template="CRITICAL FIX: Use ASYNC queries (findByRole, findByText) instead of sync queries "
                     "(getByRole, getByText) when testing lazy-loaded components.\n\n"
                     "The component is loaded via React.lazy() + Suspense. The Suspense fallback "
                     "renders first, then the lazy chunk loads asynchronously.\n\n"
                     "WRONG — sync query fails because lazy component hasn't loaded:\n"
                     "  render(<App />)\n"
                     "  screen.getByRole('heading', { name: /dashboard/i })  // FAILS — sees fallback\n\n"
                     "CORRECT — async query waits for lazy component to load:\n"
                     "  render(<App />)\n"
                     "  await screen.findByRole('heading', { name: /dashboard/i })  // WAITS for load\n\n"
                     "FIX STEPS:\n"
                     "1. READ the App/router source — check if the component is imported with React.lazy().\n"
                     "2. If lazy-loaded, ALL queries for that component's content MUST be async:\n"
                     "   - getByRole → findByRole\n"
                     "   - getByText → findByText\n"
                     "   - getByLabelText → findByLabelText\n"
                     "   - getAllByRole → findAllByRole\n"
                     "3. Make the test function async: it('...', async () => { ... })\n"
                     "4. Await every findBy* call: const heading = await screen.findByRole('heading')\n"
                     "5. For multiple queries on the same lazy component, await the FIRST one to confirm "
                     "the component has loaded, then subsequent getBy* queries are safe:\n"
                     "   await screen.findByRole('heading', { name: /dashboard/i })  // wait for load\n"
                     "   const nav = screen.getByRole('navigation')  // safe — component is loaded\n"
                     "   const table = screen.getByRole('table')  // safe — component is loaded\n"
                     "6. Alternatively, wrap assertions in waitFor():\n"
                     "   await waitFor(() => {\n"
                     "     expect(screen.getByRole('heading', { name: /dashboard/i })).toBeInTheDocument()\n"
                     "   })\n\n"
                     "SIGNS THAT A COMPONENT IS LAZY-LOADED:\n"
                     "- App.jsx has: const Dashboard = React.lazy(() => import('./pages/Dashboard'))\n"
                     "- Route element is wrapped in <Suspense>: <Suspense fallback={...}><Dashboard /></Suspense>\n"
                     "- DOM output shows only a loading message like 'Loading...' or a spinner",
        severity="error",
        tags="React.lazy,Suspense,lazy,loading,fallback,getByRole,findByRole,async,sync,query,testing-library,react,code-splitting,javascript,typescript",
    ),
    # ── Default vs Named import mismatch (Element type is invalid) ─────
    ErrorFix(
        error_type="ReactDefaultNamedImportMismatch",
        language="javascript",
        pattern=r"Element type is invalid:\s*expected a string.*but got:\s*undefined.*forgot to export.*mixed up default and named imports",
        cause="The test file uses a default import (`import Foo from './Foo'`) but the "
              "component file uses a named export (`export function Foo` or `export const Foo`), "
              "or vice versa. When the import style does not match the export style, the imported "
              "value is `undefined`, causing React to throw 'Element type is invalid'. "
              "This is the #1 most common import mistake LLMs make when generating test files — "
              "they assume a default export without reading the component source.",
        fix_template="FIX: Match the import style to the component's actual export style.\n\n"
                     "1. CHECK COMPONENT EXPORT: Read the component file being imported.\n"
                     "   - If it uses NAMED export: export function App() { ... }\n"
                     "     -> Use NAMED import: import { App } from './App'\n"
                     "   - If it uses DEFAULT export: export default function App() { ... }\n"
                     "     -> Use DEFAULT import: import App from './App'\n\n"
                     "2. ADD MISSING EXPORTS: If the component is not exported at all, add "
                     "   'export default ComponentName;' at the bottom of the file.\n\n"
                     "3. PREFER DEFAULT EXPORT: For primary components, always use 'export default' "
                     "   as per 'React Component Export Instructions' behavioral doc.\n\n"
                     "CRITICAL: NEVER assume the export style. Always READ the source file "
                     "before writing an import in a test file.",
        severity="error",
        tags="element,type,invalid,undefined,import,export,default,named,mixed,react,component,testing,javascript,typescript",
    ),
    ErrorFix(
        error_type="ThreeJsWebGLContextError",
        language="javascript",
        pattern=r"Error: Error creating WebGL context.",
        cause="Three.js WebGLRenderer requires a WebGL context which is not available in headless "
              "Node.js/JSDOM environments. Vitest and Jest tests running in JSDOM do not have a GPU.",
        fix_template="FIX: Mock the THREE.WebGLRenderer in your test or setup file.\n\n"
                     "Option 1 — Minimal Mock (Recommended):\n"
                     "  vi.mock('three', async () => {\n"
                     "    const actual = await vi.importActual('three')\n"
                     "    return {\n"
                     "      ...actual,\n"
                     "      WebGLRenderer: vi.fn().mockImplementation(() => ({\n"
                     "        setSize: vi.fn(),\n"
                     "        render: vi.fn(),\n"
                     "        dispose: vi.fn(),\n"
                     "        setClearColor: vi.fn(),\n"
                     "        domElement: document.createElement('canvas'),\n"
                     "      })),\n"
                     "    }\n"
                     "  })\n\n"
                     "Option 2 — Use vitest-canvas-mock:\n"
                     "  1. npm install --save-dev vitest-canvas-mock\n"
                     "  2. Add to vitest.config.ts: test: { setupFiles: ['vitest-canvas-mock'] }\n\n"
                     "Option 3 — Mock the global HTMLCanvasElement.prototype.getContext to return a dummy context.",
        severity="error",
        tags="threejs,webgl,renderer,canvas,mock,vitest,jest,jsdom,context,javascript,typescript",
    ),
]


# ---------------------------------------------------------------------------
# Content fix seed data
# ---------------------------------------------------------------------------

_CONTENT_FIX_SEEDS: list[ContentFix] = [
    ContentFix(
        name="tailwind-v4-directives",
        file_glob="*.css",
        content_pattern=r"^\s*@tailwind\s+(base|components|utilities)\s*;\s*$",
        replacement="",
        ensure_content='@import "tailwindcss";\n',
        flags="MULTILINE",
        collapse_blanks=True,
        language="all",
        source="core",
        description=(
            "Tailwind CSS v4 removed @tailwind base/components/utilities "
            "directives. Replace with @import \"tailwindcss\"."
        ),
    ),
    ContentFix(
        name="vitest-jest-dom-import",
        file_glob="*.test.*",
        content_pattern=r"""import\s+['"]@testing-library/jest-dom['"];?""",
        replacement="import '@testing-library/jest-dom/vitest';",
        ensure_content="",
        flags="MULTILINE",
        collapse_blanks=False,
        language="javascript",
        source="core",
        description=(
            "Replace bare @testing-library/jest-dom import with the "
            "Vitest-compatible subpath import (@testing-library/jest-dom/vitest). "
            "The bare import crashes in Vitest because expect is not global."
        ),
    ),
]


# ---------------------------------------------------------------------------
# Markdown seed data
# ---------------------------------------------------------------------------

_PATTERN_DOCS = {
    "clean-code-naming-conventions.md": {
        "title": "Clean Code Naming Conventions",
        "tags": "naming, conventions, clean-code, readability",
        "content": """## Overview

Good naming is the foundation of readable code. Names should reveal intent,
avoid disinformation, and make the code self-documenting.

## Variable Naming

### Use Intention-Revealing Names

Bad: `int d; // elapsed time in days`

Good: `int elapsedTimeInDays;`

### Avoid Disinformation

Don't use `accountList` unless it's actually a List. Prefer `accounts` or
`accountGroup` for non-list containers.

### Use Pronounceable Names

Bad: `genymdhms` (generation date, year, month, day, hour, minute, second)

Good: `generationTimestamp`

## Function Naming

### Use Verb Phrases

Functions do things — name them with verbs:
- `getUserById(id)` not `user(id)`
- `calculateTotal(items)` not `total(items)`
- `isValid()` for boolean returns

### Keep Functions Focused

If you can't name a function without using "and" or "or", it probably does
too much. Split it into separate well-named functions.

## Class Naming

### Use Noun Phrases

Classes represent things — use nouns:
- `UserRepository` not `ManageUsers`
- `PaymentProcessor` not `ProcessPayment`

### Avoid Generic Names

Avoid `Manager`, `Handler`, `Data`, `Info` unless they genuinely describe
the responsibility. Prefer domain-specific names.

## Constants

### Use Screaming Snake Case

In most languages, use `MAX_RETRY_COUNT` not `maxRetryCount` for constants.
This makes constants visually distinct from variables.
""",
    },
    "error-handling-best-practices.md": {
        "title": "Error Handling Best Practices",
        "tags": "error-handling, exceptions, resilience, best-practices",
        "content": """## Overview

Good error handling makes software robust without obscuring the main logic.
Follow these principles across all languages.

## Principle 1: Fail Fast

Validate inputs at the boundary of your system. Don't let invalid data
propagate deep into your code where failures become harder to diagnose.

```python
def process_order(order):
    if not order:
        raise ValueError("Order cannot be None")
    if order.total < 0:
        raise ValueError(f"Invalid order total: {order.total}")
    # ... proceed with valid order
```

## Principle 2: Use Specific Exception Types

Catch the most specific exception type possible. Never use bare `except:`
or `catch (Exception e)` unless you're at the top-level error boundary.

## Principle 3: Don't Swallow Exceptions

Logging an error and continuing as if nothing happened is often worse than
crashing. If you catch an exception, handle it meaningfully:

- Retry the operation (with backoff)
- Return a default value (if safe)
- Re-raise with additional context
- Translate to a domain-specific error

## Principle 4: Provide Context

Include enough context in error messages to diagnose the problem without
access to the source code:

Bad: `Error: invalid input`
Good: `Error: User ID '12345' not found in database 'users_prod'`

## Principle 5: Use Error Boundaries

Create layers where errors are caught, logged, and translated into
appropriate responses for the caller (HTTP 500, exit code 1, etc.).

## Anti-Patterns

### Pokemon Exception Handling
```
try:
    everything()
except:  # Gotta catch 'em all
    pass
```

### Error Code Returns in Exception-Based Languages
Don't return error codes when the language has exceptions. Use the
language's native error mechanism.

### Throwing Exceptions for Flow Control
Exceptions should be exceptional. Don't use try/catch for normal
branching logic.
""",
    },
    "async-patterns.md": {
        "title": "Async Programming Patterns",
        "tags": "async, concurrency, promises, patterns",
        "content": """## Overview

Asynchronous programming enables responsive applications but introduces
complexity. These patterns help manage that complexity.

## Pattern 1: Promise/Future Chaining

Chain operations that depend on each other sequentially:

```javascript
fetchUser(id)
    .then(user => fetchOrders(user.id))
    .then(orders => processOrders(orders))
    .catch(error => handleError(error));
```

## Pattern 2: Parallel Execution

Run independent operations concurrently:

```javascript
const [user, config, permissions] = await Promise.all([
    fetchUser(id),
    fetchConfig(),
    fetchPermissions(id),
]);
```

## Pattern 3: Async Iteration

Process streams of data asynchronously:

```python
async for message in websocket:
    await process_message(message)
```

## Pattern 4: Cancellation

Always support cancellation for long-running async operations:

```csharp
async Task<Data> FetchData(CancellationToken token)
{
    token.ThrowIfCancellationRequested();
    var response = await client.GetAsync(url, token);
    return await response.Content.ReadAsAsync<Data>();
}
```

## Pattern 5: Retry with Backoff

Retry transient failures with exponential backoff:

```python
async def fetch_with_retry(url, max_retries=3):
    for attempt in range(max_retries):
        try:
            return await fetch(url)
        except TransientError:
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(2 ** attempt)
```

## Anti-Patterns

### Fire and Forget
Don't start async operations without awaiting or tracking them.
Uncaught rejections crash Node.js and cause silent failures elsewhere.

### Mixing Callbacks and Promises
Pick one style and stick with it. Mixing leads to lost errors and
tangled control flow.

### Blocking the Event Loop
Never use synchronous I/O in an async context. It defeats the purpose
and blocks all concurrent operations.
""",
    },
}

_ADR_DOCS = {
    "adr-001-use-sqlite-for-vector-store.md": {
        "title": "ADR-001: Use SQLite for Vector Store",
        "tags": "adr, sqlite, vector-store, architecture",
        "content": """## Status

Accepted

## Context

The AgentChanti knowledge base requires a vector store for semantic search
over code symbols and documentation. We evaluated several options:

1. **SQLite** — Lightweight embedded database with custom vector support
2. **ChromaDB** — Lightweight embedded vector store
3. **FAISS** — Facebook's similarity search library (no server)
4. **Pinecone** — Cloud-hosted managed vector database

## Decision

We chose **SQLite** for the following reasons:

### Advantages
- **Zero-dependency**: No external server or Docker required
- **Embedded**: Runs in-process, no network overhead
- **Portable**: Single-file database, easy to backup and distribute
- **Battle-tested**: SQLite is the most widely deployed database
- **Simple operations**: No container management needed

### Trade-offs
- Less feature-rich than dedicated vector databases
- Custom cosine similarity implementation via NumPy
- No built-in clustering or HNSW index

## Consequences

- All vector operations use the local SQLite database
- Embeddings stored as binary blobs with cosine similarity search
- No external services required — fully offline operation
- Collection naming convention: `local_{project_slug}` for per-project,
  `global_kb` for the shared knowledge base
""",
    },
    "adr-002-tree-sitter-for-ast-parsing.md": {
        "title": "ADR-002: Use Tree-sitter for AST Parsing",
        "tags": "adr, tree-sitter, parsing, ast, architecture",
        "content": """## Status

Accepted

## Context

Building a code graph requires parsing source code into an AST across
multiple languages. We evaluated:

1. **Tree-sitter** — Incremental parsing library with grammars for 100+ languages
2. **Language-specific parsers** — ast (Python), @babel/parser (JS), etc.
3. **ctags/cscope** — Symbol-level indexing without full AST
4. **LSP servers** — Language Server Protocol for per-language intelligence

## Decision

We chose **Tree-sitter** for the following reasons:

### Advantages
- **Multi-language**: One parser framework for Python, JavaScript, TypeScript,
  Java, Go, Rust, C, C++, Ruby, PHP, C# — and growing
- **Incremental parsing**: Re-parses only changed portions, enabling fast
  watch-mode updates
- **Concrete syntax tree**: Preserves all source text, enabling accurate
  line-number tracking and code extraction
- **Widely adopted**: Used by GitHub, Neovim, Zed, and Helix editors

### Trade-offs
- Grammars need per-language installation (`tree-sitter-python`, etc.)
- Less semantic depth than dedicated LSP servers (no type inference)
- Some grammars have edge cases with advanced language features

## Consequences

- The parser module wraps Tree-sitter for all supported languages
- Each language grammar is an optional dependency
- Symbol extraction traverses the Tree-sitter CST to find functions,
  classes, imports, and call sites
- The code graph (NetworkX DiGraph) is built from extracted symbols
""",
    },
}

_DOC_DOCS = {
    "tree-sitter-usage-guide.md": {
        "title": "Tree-sitter Usage Guide",
        "tags": "tree-sitter, parsing, guide, setup",
        "content": """## Overview

Tree-sitter is the core parsing engine for the AgentChanti local knowledge
base. This guide covers setup, usage, and troubleshooting.

## Installation

Install tree-sitter and the language grammars you need:

```bash
pip install tree-sitter
pip install tree-sitter-python tree-sitter-javascript tree-sitter-typescript
pip install tree-sitter-java tree-sitter-go tree-sitter-rust
```

## How It Works

### Parsing Flow
1. Source code is loaded as bytes
2. Tree-sitter parses it into a Concrete Syntax Tree (CST)
3. AgentChanti traverses the CST to extract symbols
4. Symbols are added to the NetworkX code graph

### Symbol Types Extracted
- **Functions/Methods**: name, parameters, return type, docstring, body
- **Classes**: name, base classes, docstring, method list
- **Imports**: module name, imported names
- **Variables**: module-level assignments

## Custom Queries

Tree-sitter supports S-expression queries for targeted extraction:

```scheme
;; Find all function definitions
(function_definition
  name: (identifier) @function.name
  parameters: (parameters) @function.params)
```

## Troubleshooting

### Grammar Not Found
Ensure the language grammar package is installed:
```bash
pip install tree-sitter-{language}
```

### Parse Errors
Tree-sitter is error-tolerant — it produces a partial tree even with
syntax errors. Check for `ERROR` nodes in the tree to find problem areas.

### Performance
For large files (>10K lines), parsing may take >100ms. The incremental
parser helps by only re-parsing changed regions during watch mode.
""",
    },
    "vector-store-usage-guide.md": {
        "title": "Vector Store Usage Guide",
        "tags": "sqlite, vector-store, setup, guide",
        "content": """## Overview

AgentChanti uses a local SQLite-based vector store for semantic search
over code symbols and documentation. No external services are required.

## Prerequisites

- Python 3.9+
- NumPy (optional, for optimized similarity search)

## Quick Start

```bash
# Index the project
agentchanti kb index

# Embed symbols into the vector store
agentchanti kb embed

# Search
agentchanti kb search "authentication middleware"
```

## How It Works

1. Code symbols are extracted via Tree-sitter
2. Embeddings are generated using the configured LLM provider
3. Embeddings are stored in a local SQLite database
4. Searches compute cosine similarity against stored embeddings

## Storage

Vector data is stored at:
- Per-project: `{project}/.agentchanti/kb/local/vectors.db`
- Global: `~/.agentchanti/global_kb/vectors.db`

## Troubleshooting

### No Results
Ensure you've run indexing first: `agentchanti kb index && agentchanti kb embed`

### Slow Search
Install NumPy for optimized vector operations: `pip install numpy`
""",
    },
    "testing-library-errors-guide.md": {
        "title": "Testing Library Common Errors Guide",
        "tags": "testing-library, react, errors, testing, guide",
        "content": """## Overview

React Testing Library helps you test your UI in a user-centric way. However, tests often fail with `TestingLibraryElementError` due to mismatched roles, incorrect assumptions about the rendered DOM, or async timing issues. This guide helps you properly fix these errors.

## 1. Unable to find role="X" and name "Y"

When you encounter:
`TestingLibraryElementError: Unable to find role="heading" and name /pricing/i`

### Root Cause
You assume an element exists with a specific HTML Tag role (`h1` -> `heading`, `a` -> `link`, `button`) but the actual component source renders a different tag (e.g. `p` instead of `h2`) or uses slightly different text.

### How to Fix
- Read the component source code to see what HTML element is used.
- Check the error's `DOM` output. It gives you the actual rendered tags.
- Use `screen.debug()` in the test if needed.
- If it's a `div` or `span`, consider using `screen.getByText(/pricing/i)` instead. `getByRole` usually requires explicit semantic HTML tags.

## 2. Unable to find an accessible element with the role...

This means the test assumed an ARIA attribute exist that doesn't.
For example, a `<section>` without an `aria-label` does NOT expose the 'region' role. Wait until you have read the actual source component before adding roles to your test queries.

## 3. Empty DOM (`<body><div></div></body>`)

If your query times out with an empty DOM, this usually means the component failed to render.
- Is it missing a `Provider`? e.g., missing `<MemoryRouter>`?
- Is it lazy loaded? Use `await screen.findByRole` instead of `screen.getByRole`.

## 4. Function Matchers for complex Strings

For text spread across multiple elements, text matches can be tricky.
Instead of: `expect(screen.getByText('User Profile: Uday')).toBeInTheDocument()`
Use:
```js
expect(screen.getByText((content, element) => {
   return element.textContent.includes('User Profile: Uday');
})).toBeInTheDocument();
```
""",
    },
    "vitest-react-testing-setup.md": {
        "title": "Vitest React Testing Library Setup — Required Packages",
        "tags": "vitest, react, testing-library, jsdom, setup, packages, npm, install, required, jest-dom",
        "content": """## Overview

When setting up a React project with Vitest and @testing-library/react, the following
packages are ALL required. Missing any of these causes hard-to-debug import/runtime errors.

## Required Dev Packages (complete list)

```bash
npm install --save-dev vitest jsdom @testing-library/react @testing-library/dom @testing-library/jest-dom @testing-library/user-event @vitejs/plugin-react
```

### Package purposes:
| Package | Purpose | Error if missing |
|---------|---------|-----------------|
| `vitest` | Test runner | `vitest: command not found` |
| `jsdom` | DOM environment for tests | `Error: Failed to find a valid JSDOM implementation` or `ReferenceError: document is not defined` |
| `@testing-library/react` | `render()`, `screen` queries | `Cannot find module '@testing-library/react'` |
| `@testing-library/dom` | Core DOM queries (peer dep of @testing-library/react) | `Cannot find module '@testing-library/dom'` |
| `@testing-library/jest-dom` | Custom matchers: `toBeInTheDocument()`, `toHaveClass()` | `TypeError: expect(...).toBeInTheDocument is not a function` |
| `@testing-library/user-event` | Realistic user interactions: `userEvent.click()` | `Cannot find module '@testing-library/user-event'` |
| `@vitejs/plugin-react` | JSX transform for Vite | `[plugin:vite:esbuild] Failed to parse source for import analysis` |

## Required vitest.config.js

```js
import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: './vitest.setup.js',
  },
})
```

## Required vitest.setup.js

```js
import '@testing-library/jest-dom/vitest'
```

**CRITICAL**: Use `@testing-library/jest-dom/vitest` (NOT `@testing-library/jest-dom`).
The base import only works with Jest. The `/vitest` subpath registers matchers correctly
with Vitest's `expect`.

## Common mistakes

1. **Missing `jsdom`**: Vitest defaults to `node` environment. Without `jsdom`, there is
   no `document`, `window`, or DOM — all component renders fail silently or crash.
2. **Missing `@testing-library/dom`**: This is a peer dependency of `@testing-library/react`.
   npm v7+ auto-installs peer deps, but older versions or CI environments may not.
3. **Wrong jest-dom import**: `import '@testing-library/jest-dom'` in Vitest causes
   `expect.extend is not a function` or matchers not being registered.
4. **Missing `@vitejs/plugin-react`**: Without the React plugin, Vite cannot transform
   JSX in `.jsx`/`.tsx` files, causing parse errors in tests.
""",
    },
    "threejs-webgl-error-fix.md": {
        "title": "Three.js + Vitest: Fixing WebGL Context Errors",
        "tags": "threejs, webgl, vitest, jest, jsdom, mock, canvas, renderer, setup",
        "content": """## Overview

When testing components that use Three.js (like `WebGLRenderer`) in a headless environment 
(Vitest or Jest with JSDOM), you will often encounter the error:
`Error: Error creating WebGL context.`

This happens because JSDOM does not implement a WebGL context.

## Solution 1: Mocking WebGLRenderer (Recommended)

The most efficient way to fix this in component tests is to mock the `WebGLRenderer` 
to prevent it from trying to initialize a real WebGL context.

### Using Vitest

Add this to your test file or `vitest.setup.js`:

```javascript
import { vi } from 'vitest'

vi.mock('three', async () => {
  const actual = await vi.importActual('three')
  return {
    ...actual,
    WebGLRenderer: vi.fn().mockImplementation(() => ({
      setSize: vi.fn(),
      setPixelRatio: vi.fn(),
      render: vi.fn(),
      dispose: vi.fn(),
      setClearColor: vi.fn(),
      domElement: document.createElement('canvas'),
    })),
  }
})
```

## Solution 2: Automated Canvas Mocking

If you have many components using Canvas/WebGL, you can use `vitest-canvas-mock`.

1. **Install**:
```bash
npm install --save-dev vitest-canvas-mock
```

2. **Configure `vitest.config.ts`**:
```typescript
export default defineConfig({
  test: {
    environment: 'jsdom',
    setupFiles: ['vitest-canvas-mock'],
  },
})
```

## Solution 3: Manual Context Mocking

If you only need to bypass the context creation check, you can mock `getContext` globally:

```javascript
HTMLCanvasElement.prototype.getContext = vi.fn().mockReturnValue({
  // Mock minimal context methods if needed
  fillRect: vi.fn(),
  clearRect: vi.fn(),
  getImageData: vi.fn(),
  putImageData: vi.fn(),
  createImageData: vi.fn(),
  setTransform: vi.fn(),
  drawImage: vi.fn(),
  save: vi.fn(),
  restore: vi.fn(),
  beginPath: vi.fn(),
  moveTo: vi.fn(),
  lineTo: vi.fn(),
  closePath: vi.fn(),
  stroke: vi.fn(),
  translate: vi.fn(),
  scale: vi.fn(),
  rotate: vi.fn(),
  arc: vi.fn(),
  fill: vi.fn(),
  measureText: vi.fn().mockReturnValue({ width: 0 }),
  transform: vi.fn(),
  rect: vi.fn(),
  clip: vi.fn(),
})
```

## Verification

After applying one of these fixes, your Three.js components should render without 
throwing the WebGL context error, allowing you to test other aspects of the component 
(props, headings, UI elements).
""",
    },
}

_BEHAVIORAL_DOCS = {
    "code-review-instructions.md": {
        "title": "Code Review Instructions",
        "tags": "code-review, instructions, behavioral, quality",
        "content": """## Overview

When performing code review, follow these structured instructions to
ensure consistent, thorough, and constructive feedback.

## Review Checklist

### 1. Correctness
- Does the code do what it's supposed to do?
- Are edge cases handled?
- Are there potential null/undefined access issues?
- Are error conditions handled appropriately?

### 2. Security
- Is user input validated and sanitized?
- Are there SQL injection or XSS vulnerabilities?
- Are secrets hardcoded?
- Are permissions checked appropriately?

### 3. Performance
- Are there N+1 query patterns?
- Are expensive operations cached when appropriate?
- Are there unnecessary allocations in hot paths?
- Could data structures be more efficient?

### 4. Readability
- Are variable and function names descriptive?
- Is the code self-documenting or well-commented?
- Are functions at a single level of abstraction?
- Is the control flow easy to follow?

### 5. Maintainability
- Does the code follow existing patterns in the codebase?
- Are there duplicated code blocks that should be extracted?
- Is the code testable?
- Are dependencies minimized?

## Giving Feedback

### Be Specific
Bad: "This function is too complex"
Good: "This function has 3 levels of nesting — extract the inner loop into a helper"

### Explain Why
Don't just say what to change — explain why the change matters.

### Distinguish Severity
- **Blocker**: Must fix before merge (bugs, security issues)
- **Suggestion**: Recommended improvement (naming, structure)
- **Nit**: Minor style preference (formatting, comment wording)
""",
    },
    "error-analysis-instructions.md": {
        "title": "Error Analysis Instructions",
        "tags": "error-analysis, debugging, instructions, behavioral",
        "content": """## Overview

When analyzing errors, follow this systematic approach to identify root
causes and suggest effective fixes.

## Step 1: Classify the Error

Determine the error category:
- **Syntax Error**: Code doesn't parse — missing brackets, typos
- **Type Error**: Wrong data type for an operation
- **Runtime Error**: Crash during execution — null access, index bounds
- **Logic Error**: Code runs but produces wrong results
- **Resource Error**: File not found, connection refused, timeout

## Step 2: Read the Full Stack Trace

- Start from the **bottom** of the stack trace (the actual error)
- Work **upward** to find the originating call in user code
- Identify if the error is in user code or library code
- Note the file, line number, and function name

## Step 3: Identify Root Cause

Common root causes:
- **Missing null check**: Object is None/null/nil when accessed
- **Wrong assumption about data**: Expected format differs from actual
- **Race condition**: Concurrent access to shared state
- **State management**: Component lifecycle or state machine error
- **Configuration**: Wrong environment, missing env vars, wrong paths

## Step 4: Suggest a Fix

A good fix suggestion includes:
1. **What to change**: The specific code modification
2. **Why it fixes the issue**: Connect the change to the root cause
3. **How to prevent recurrence**: Tests, type guards, validation

## Step 5: Suggest Preventive Measures

- Add unit tests that reproduce the error
- Add type annotations/guards at the boundary
- Improve error messages for faster future diagnosis
- Consider if similar bugs could exist elsewhere
""",
    },
    "react-component-test-generation-instructions.md": {
        "title": "React Component Test Generation Instructions",
        "tags": "react, testing, vitest, jest, component, test-generation, behavioral, testing-library, getByText, getByRole, within, render, instructions",
        "content": """## Overview

When generating tests for React components, you MUST follow these rules to avoid
the most common test failures. These rules apply to both Vitest and Jest projects
using @testing-library/react.

## Rule 1: READ the source component BEFORE writing any assertion

NEVER assume what a component renders. Before writing any `getByText`, `getByRole`,
or any query, read the actual component source to determine:
- Exact text content rendered (including text from props/state)
- Element types used (button, a, span, div, h1, etc.)
- ARIA roles and accessible names
- How many instances of each element exist
- Whether text is split across multiple DOM elements

## Rule 2: Avoid `getByText` with `textContent.includes()` function matchers

This is the #1 source of "Found multiple elements" errors.

**WHY IT FAILS**: `element.textContent` includes ALL descendant text. When you write:
```js
screen.getByText((content, element) => element?.textContent?.includes('Jane Doe'))
```
Every ancestor of the element containing "Jane Doe" ALSO matches, because parent
elements' textContent includes their children's text. This causes `getByText` to
find multiple matches and throw.

**CORRECT APPROACHES** (in order of preference):
```js
// 1. BEST: Use getByRole with accessible name — no ambiguity
screen.getByRole('heading', { name: /jane doe/i })

// 2. Use getByText with plain string or regex (matches direct text content)
screen.getByText(/Jane Doe/)

// 3. If you MUST use a function matcher, filter by tag to avoid parent matches
screen.getByText((content, element) => {
  return element?.tagName === 'SPAN' && element?.textContent?.includes('Jane Doe')
})

// 4. Use getAllByText and take the first match
const matches = screen.getAllByText((content, element) => {
  return element?.textContent?.includes('Jane Doe')
})
expect(matches[0]).toBeInTheDocument()

// 5. Scope with within() to a specific container
import { within } from '@testing-library/react'
const header = screen.getByRole('banner')
within(header).getByText(/Jane Doe/)
```

## Rule 3: Prefer `getByRole` over `getByText`

`getByRole` is more robust because it queries by ARIA role + accessible name, not
raw text. It avoids issues with split text, duplicate text, and structural changes.

```js
// BAD — fragile, breaks when text content changes slightly
screen.getByText('Submit Order')

// GOOD — resilient to text changes, tests accessibility too
screen.getByRole('button', { name: /submit/i })

// BAD — matches all links on the page
screen.getByText('Home')

// GOOD — scoped to navigation
const nav = screen.getByRole('navigation')
within(nav).getByRole('link', { name: /home/i })
```

## Rule 4: Use `within()` when the page has duplicate elements

Dashboards, layouts, and pages with headers/footers/sidebars commonly have multiple
elements with the same text or role (e.g., "Home" link in both nav and footer).

```js
import { within } from '@testing-library/react'

// Scope queries to a specific section
const sidebar = screen.getByRole('navigation')
within(sidebar).getByRole('link', { name: /dashboard/i })

const main = screen.getByRole('main')
within(main).getByRole('heading', { name: /welcome/i })
```

## Rule 5: Use `getAllBy*` when multiple matches are expected

If the component intentionally renders multiple matching elements (e.g., a list of
cards, table rows, repeated links), use `getAllBy*` and assert on the array:

```js
const cards = screen.getAllByRole('article')
expect(cards).toHaveLength(3)

const links = screen.getAllByRole('link', { name: /details/i })
expect(links.length).toBeGreaterThanOrEqual(1)
```

## Rule 6: Always wrap routed components in MemoryRouter

Any component that uses React Router hooks (useLocation, useNavigate, useParams)
or components (Link, NavLink, Outlet) MUST be wrapped in MemoryRouter:

```js
import { MemoryRouter } from 'react-router-dom'

render(
  <MemoryRouter initialEntries={['/dashboard']}>
    <Dashboard />
  </MemoryRouter>
)
```

## Rule 7: Never hardcode expected values from assumption

- Count elements in the source data/props to determine expected `.toHaveLength()`
- Read the component to find exact text, class names, and roles
- If the component uses props/context for text, mock those values and assert on
  the mocked values — not on values you invented

## Rule 8: Use `queryBy*` for elements that may not exist

`getBy*` throws if the element is not found. Use `queryBy*` when testing that
something is NOT rendered, or when the element is conditionally rendered:

```js
// Testing absence
expect(screen.queryByText(/error/i)).not.toBeInTheDocument()

// Conditionally rendered elements
const modal = screen.queryByRole('dialog')
if (modal) {
  expect(within(modal).getByText(/confirm/i)).toBeInTheDocument()
}
```

## Rule 9: Import jest-dom correctly for Vitest

```js
// WRONG — crashes in Vitest
import '@testing-library/jest-dom'

// CORRECT — Vitest-compatible subpath
import '@testing-library/jest-dom/vitest'
```

## Rule 10: Use correct file extensions for JSX

Vite CANNOT parse JSX in `.js` or `.ts` files. If your test contains JSX
(`<Component />`, `render(<App />)`), the test file MUST use:
- `.test.jsx` (for JavaScript)
- `.test.tsx` (for TypeScript)

## Rule 11: NEVER assume ARIA roles or labels — check the source

This is the #1 cause of "Unable to find an accessible element" errors. Most
components use plain HTML elements WITHOUT explicit ARIA attributes. You MUST
check the source code before using `getByRole('region')`, `getByLabelText()`,
or any role-based query with an accessible name.

### HTML Implicit Roles Reference

These elements have implicit ARIA roles ONLY under certain conditions:

| HTML Element | ARIA Role | Condition |
|-------------|-----------|-----------|
| `<section aria-label="...">` | `region` | **ONLY when it has an accessible name** (aria-label or aria-labelledby) |
| `<section>` (no label) | **NO role** | Not findable by `getByRole('region')` |
| `<div>` | **NO role** | Never has an implicit role |
| `<span>` | **NO role** | Never has an implicit role |
| `<nav>` | `navigation` | Always (no label required) |
| `<main>` | `main` | Always |
| `<header>` | `banner` | When not nested in `<article>`, `<section>`, etc. |
| `<footer>` | `contentinfo` | When not nested in `<article>`, `<section>`, etc. |
| `<form aria-label="...">` | `form` | **ONLY when it has an accessible name** |
| `<form>` (no label) | **NO role** | Not findable by `getByRole('form')` |
| `<table>` | `table` | Always |
| `<button>` | `button` | Always |
| `<a href="...">` | `link` | Only when it has an `href` attribute |
| `<input type="text">` | `textbox` | Always |
| `<h1>`-`<h6>` | `heading` | Always |
| `<ul>`, `<ol>` | `list` | Always |
| `<li>` | `listitem` | Always |
| `<img alt="...">` | `img` | When it has a non-empty `alt` |

### What to do instead of assuming ARIA roles

```js
// BAD: assumes <section> has aria-label="stats" — will fail if it doesn't
screen.getByRole('region', { name: /stats/i })

// BAD: assumes any element has aria-label="stats"
screen.getByLabelText(/stats/i)

// GOOD: verify the section's CONTENT is rendered instead
expect(screen.getByText(/total users/i)).toBeInTheDocument()
expect(screen.getByText(/revenue/i)).toBeInTheDocument()

// GOOD: use heading if the section has one
screen.getByRole('heading', { name: /statistics/i })

// GOOD: use elements that ALWAYS have roles (nav, main, header, footer, table)
screen.getByRole('navigation')
screen.getByRole('main')
screen.getByRole('table')

// GOOD: scope to <main> then check content inside
const main = screen.getByRole('main')
expect(within(main).getByText(/total users/i)).toBeInTheDocument()
```

### Decision Tree for querying a section

1. Does the source have `<section aria-label="...">` or `role="region"`? → Use `getByRole('region', { name: ... })`
2. Does the section have a heading (`<h2>`, `<h3>`, etc.)? → Use `getByRole('heading', { name: ... })`
3. Does the section have unique text content? → Use `getByText(...)` on the content
4. Is the section inside `<main>`? → Use `within(screen.getByRole('main')).getByText(...)`
5. Does the source have `data-testid`? → Use `getByTestId(...)`
6. None of the above? → Assert on the specific content the section renders

## Rule 12: NEVER assume async/loading content inside `waitFor`

`waitFor` re-runs its callback until it passes or times out. If you assert on text
the component NEVER renders, `waitFor` hangs until timeout and the test fails.

**WRONG — assumes component renders "No records found." after loading:**
```js
await waitFor(() => {
  screen.getByText((c, el) => el.tagName === 'P' && c === 'No records found.')
})
```

**CORRECT — read the source to find what ACTUALLY renders after loading:**
```js
// 1. Read the source component to find its actual loading/loaded/empty states
// 2. Assert ONLY on text or elements you verified exist in the source
await waitFor(() => {
  // Use actual text from the component's return/render JSX
  screen.getByRole('table')  // if it renders a table after loading
})
```

**Steps for async component tests:**
1. READ the source component to find:
   - What renders during loading (spinner, skeleton, "Loading..." text)
   - What renders after loading completes (tables, cards, lists)
   - What renders for empty/error states — look for actual JSX like `<p>No data</p>`
2. First assert the loading indicator appears (if any)
3. Then `waitFor` the ACTUAL post-loading content
4. NEVER invent text content for empty states — the component may not have one
5. If unsure what renders, use `screen.debug()` in development to inspect

## Rule 13: NEVER wrap App in MemoryRouter if App has its own Router

If the App component already contains `<BrowserRouter>` or `<HashRouter>`, wrapping
it in `<MemoryRouter>` causes: "You cannot render a <Router> inside another <Router>"

**READ the App source first.** Check if it imports and uses BrowserRouter/HashRouter.

```js
// WRONG — App already has BrowserRouter internally
render(
  <MemoryRouter initialEntries={['/']}>
    <App />
  </MemoryRouter>
)

// CORRECT Option A — Mock BrowserRouter to become MemoryRouter
vi.mock('react-router-dom', async () => {
  const actual = await vi.importActual('react-router-dom')
  return {
    ...actual,
    BrowserRouter: ({ children }) => (
      <actual.MemoryRouter initialEntries={['/']}>{children}</actual.MemoryRouter>
    ),
  }
})
render(<App />)

// CORRECT Option B — Test page components individually, not App
render(
  <MemoryRouter initialEntries={['/dashboard']}>
    <Routes>
      <Route path="/dashboard" element={<Dashboard />} />
    </Routes>
  </MemoryRouter>
)
```

## Rule 14: Use ASYNC queries for lazy-loaded (React.lazy) components

When a component is loaded via `React.lazy()` + `<Suspense>`, it does NOT render
immediately. The Suspense fallback renders first (e.g. "Loading dashboard..."), then
the lazy chunk loads asynchronously. **Sync queries (`getByRole`, `getByText`) will
only see the fallback and FAIL.**

**How to detect lazy loading:**
- App.jsx has: `const Dashboard = React.lazy(() => import('./pages/Dashboard'))`
- Route uses `<Suspense fallback={<p>Loading...</p>}><Dashboard /></Suspense>`
- DOM output in test error shows only a loading message, not the real content

**WRONG — sync query runs before lazy component loads:**
```js
render(<App />)
screen.getByRole('heading', { name: /dashboard/i })  // FAILS — DOM has "Loading..."
screen.getByRole('navigation')  // FAILS — DOM has "Loading..."
screen.getByRole('table')  // FAILS — DOM has "Loading..."
```

**CORRECT — use findBy* (async) to wait for lazy component:**
```js
render(<App />)
// findBy* returns a Promise that resolves when the element appears
await screen.findByRole('heading', { name: /dashboard/i })  // waits for load

// After the first findBy* resolves, the component is loaded.
// Subsequent queries can be sync:
const nav = screen.getByRole('navigation')
const table = screen.getByRole('table')
```

**Conversion rules:**
| Sync (WRONG for lazy) | Async (CORRECT for lazy) |
|---|---|
| `getByRole(...)` | `await findByRole(...)` |
| `getByText(...)` | `await findByText(...)` |
| `getByLabelText(...)` | `await findByLabelText(...)` |
| `getAllByRole(...)` | `await findAllByRole(...)` |

**Key pattern:** Await the FIRST query to confirm the lazy component loaded,
then use sync queries for the rest of the component's elements.

## Rule 15: Diagnose "Empty Root" symptoms first

If a test fails with "Unable to find an element" and the DOM output in the error message
shows an empty body (e.g., `<body> <div> </div> </body>` or `<body> </body>`):

**DO NOT** simply try to change the query or add a function matcher. An empty DOM means
the component is not rendering at all.

**FIX FLOW FOR EMPTY DOM:**
1. **Check for missing context**: If the component uses Router/Theme hooks, it might
   return null if not wrapped in a Provider. Wrap in `<MemoryRouter>` or `<ThemeProvider>`.
2. **Check for lazy loading**: If the component is lazy-loaded, you MUST use `findBy*`
   (async) queries. The first render of a lazy component is usually empty or fallback.
3. **Check imports**: If you imported a component as `undefined` (mixed up default/named
   imports), React might render nothing.
4. **Check props**: Ensure you are passing mandatory props that, if missing, might
   cause the component to return null.

## Rule 16: Use `toHaveClass` for CSS assertions

**NEVER** use regex on `className` to assert presence/absence of CSS classes. Tailwind
classes with hyphens (`-translate-x-full`) or colons (`md:flex`) are notoriously difficult
to match correctly with regex due to word boundary (`\b`) behavior and escaping requirements.

**WRONG (brittle):**
```js
expect(nav.className).toMatch(/\b-translate-x-full\b/)  // Fails: \b doesn't match before hyphen
expect(nav.className).toMatch(/md\\:flex/)              // Fails: complex escaping needed
```

**CORRECT (robust):**
```js
import '@testing-library/jest-dom/vitest' // ensure matchers are registered

expect(nav).toHaveClass('-translate-x-full')
expect(nav).toHaveClass('md:flex')
expect(nav).not.toHaveClass('hidden')
```
""",
    },
    "react-export-default-instructions.md": {
        "title": "React Component Export Instructions",
        "tags": "react, jsx, tsx, export, default, naming, consistency, behavioral, instructions, component, create, modify, edit, generate",
        "content": """## CRITICAL: React Component Export Rules

When generating or modifying React JSX/TSX component files, you MUST follow these rules.
Violating these rules WILL break the application at runtime with missing default export errors.

## Rule 1: ALWAYS include `export default` for the primary component

Every React component file (.jsx/.tsx) MUST have exactly one `export default` for its primary component.
This is MANDATORY — Vite, Next.js, React Router, and most frameworks require default exports for page/route components.

CORRECT examples:
```jsx
// Option A: inline default export
export default function Dashboard() {
  return <div>Dashboard</div>
}

// Option B: separate default export
function Dashboard() {
  return <div>Dashboard</div>
}
export default Dashboard;
```

WRONG — missing export default (WILL cause runtime error):
```jsx
// BAD: no default export — other files importing this will get undefined
function Dashboard() {
  return <div>Dashboard</div>
}
```

## Rule 2: NEVER remove existing `export default` statements

When editing or modifying a component file, you MUST preserve the `export default` statement.
If the file already has `export default`, it MUST remain in your output. Removing it breaks
every file that imports this component. This is the #1 most common LLM mistake when editing
React components — the LLM rewrites the component but drops the export default line.

## Rule 3: When rewriting a component, put `export default` at the END of the file

If you rewrite the entire component, always include `export default ComponentName;` as the
last line, or use `export default function ComponentName()` at the function declaration.

## Rule 4: Ensure consistency between component name and export

The default-exported component name should match the filename in PascalCase:
- `Dashboard.jsx` → `export default function Dashboard()`
- `UserProfile.jsx` → `export default function UserProfile()`

## Rule 5: Match import style in tests and other files

Default export → default import: `import Dashboard from './Dashboard'`
Named export → named import: `import { Dashboard } from './Dashboard'`

## Rule 6: Avoid mixed default and named exports for the same component

Use `export default` for the main component. Named exports for helpers/constants only.
""",
    },
}


# ---------------------------------------------------------------------------
# Markdown chunk helpers
# ---------------------------------------------------------------------------

def _chunk_markdown(text: str, title: str, min_size: int = 100, max_size: int = 1500) -> list[str]:
    """
    Split markdown text into chunks by heading sections.

    Parameters
    ----------
    text:
        The markdown body (without frontmatter).
    title:
        Title to prepend to every chunk for context.
    min_size:
        Merge sections smaller than this.
    max_size:
        Split sections larger than this.

    Returns
    -------
    list[str]
        List of text chunks.
    """
    import re as _re
    sections: list[str] = []
    current: list[str] = []

    for line in text.split("\n"):
        if _re.match(r"^#{2,3}\s+", line) and current:
            sections.append("\n".join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        sections.append("\n".join(current))

    # Merge small sections
    merged: list[str] = []
    buf = ""
    for sec in sections:
        if len(buf) + len(sec) < min_size:
            buf = buf + "\n" + sec if buf else sec
        else:
            if buf:
                merged.append(buf)
            buf = sec
    if buf:
        merged.append(buf)

    # Split oversized chunks
    final: list[str] = []
    for chunk in merged:
        if len(chunk) <= max_size:
            final.append(f"Title: {title}\n\n{chunk.strip()}")
        else:
            # Split at paragraph boundaries
            paragraphs = chunk.split("\n\n")
            sub_buf = ""
            for para in paragraphs:
                if len(sub_buf) + len(para) > max_size and sub_buf:
                    final.append(f"Title: {title}\n\n{sub_buf.strip()}")
                    sub_buf = para
                else:
                    sub_buf = sub_buf + "\n\n" + para if sub_buf else para
            if sub_buf:
                final.append(f"Title: {title}\n\n{sub_buf.strip()}")

    return final


# ---------------------------------------------------------------------------
# File writers
# ---------------------------------------------------------------------------

def _write_md_file(
    directory: str,
    filename: str,
    title: str,
    category: str,
    tags: str,
    language: str,
    content: str,
) -> str:
    """Write a markdown file with frontmatter.  Returns the absolute path."""
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, filename)
    frontmatter = (
        "---\n"
        f'title: "{title}"\n'
        f'category: "{category}"\n'
        f'tags: "{tags}"\n'
        f'language: "{language}"\n'
        f'version: "1.0.0"\n'
        f'created_at: "2025-01-01"\n'
        f'source: "seeder"\n'
        "---\n\n"
    )
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(frontmatter + content)
    return path


# ---------------------------------------------------------------------------
# Main seeder
# ---------------------------------------------------------------------------

def seed(
    embed: bool = True,
    project_root: Optional[str] = None,
    api_client=None,
    base_dir: Optional[str] = None,
) -> dict:
    """
    Seed the global knowledge base with sample data.

    Parameters
    ----------
    embed:
        If True, embed markdown documents into the SQLite vector store.
    project_root:
        Project root for vector store path.  Defaults to cwd.
    api_client:
        LLM client to use for embedding.

    Returns
    -------
    dict
        Summary with keys: errors_seeded, docs_seeded, chunks_embedded.
    """
    project_root = project_root or os.getcwd()
    # Allow callers (tests) to redirect all I/O to an isolated directory.
    core_dir = os.path.join(base_dir, "core") if base_dir else _CORE_DIR
    registry_dir = os.path.join(base_dir, "registry") if base_dir else _REGISTRY_DIR
    if base_dir:
        os.makedirs(core_dir, exist_ok=True)
    summary = {
        "errors_seeded": 0,
        "content_fixes_seeded": 0,
        "docs_seeded": 0,
        "chunks_embedded": 0,
    }

    # ── 1a. Seed errors.db ──────────────────────────────────────────────
    db_path = os.path.join(core_dir, "errors.db")
    edict = ErrorDict(db_path)
    # Re-seed when the seed list has grown (new entries added to code).
    # This ensures new KB error fixes are picked up without requiring
    # users to manually delete errors.db.
    current_count = edict.count()
    if current_count == 0:
        edict.bulk_insert(_ERROR_SEEDS)
    elif current_count < len(_ERROR_SEEDS):
        logger.info(
            "errors.db has %d records but seed list has %d — re-seeding",
            current_count, len(_ERROR_SEEDS),
        )
        edict.clear()
        edict.bulk_insert(_ERROR_SEEDS)
    else:
        logger.debug("errors.db already populated (%d records), skipping re-seed",
                     current_count)
    summary["errors_seeded"] = edict.count()
    logger.info("Seeded %d errors into %s", summary["errors_seeded"], db_path)

    # ── 1b. Seed content fixes ───────────────────────────────────────────
    current_cf_count = edict.count_content_fixes()
    if current_cf_count == 0:
        edict.bulk_insert_content_fixes(_CONTENT_FIX_SEEDS)
    elif current_cf_count < len(_CONTENT_FIX_SEEDS):
        logger.info(
            "content_fixes has %d records but seed list has %d — re-seeding",
            current_cf_count, len(_CONTENT_FIX_SEEDS),
        )
        edict.clear_content_fixes()
        edict.bulk_insert_content_fixes(_CONTENT_FIX_SEEDS)
    else:
        logger.debug("content_fixes already populated, skipping re-seed")
    summary["content_fixes_seeded"] = edict.count_content_fixes()
    logger.info("Seeded %d content fixes into %s",
                summary["content_fixes_seeded"], db_path)

    # ── 2. Write markdown files ─────────────────────────────────────────
    md_files: list[tuple[str, str, str]] = []  # (path, category, title)

    for filename, meta in _PATTERN_DOCS.items():
        path = _write_md_file(
            os.path.join(registry_dir, "patterns"),
            filename,
            meta["title"],
            "pattern",
            meta["tags"],
            "all",
            meta["content"],
        )
        md_files.append((path, "pattern", meta["title"]))

    for filename, meta in _ADR_DOCS.items():
        path = _write_md_file(
            os.path.join(registry_dir, "adrs"),
            filename,
            meta["title"],
            "adr",
            meta["tags"],
            "all",
            meta["content"],
        )
        md_files.append((path, "adr", meta["title"]))

    for filename, meta in _DOC_DOCS.items():
        path = _write_md_file(
            os.path.join(registry_dir, "docs"),
            filename,
            meta["title"],
            "doc",
            meta["tags"],
            "all",
            meta["content"],
        )
        md_files.append((path, "doc", meta["title"]))

    for filename, meta in _BEHAVIORAL_DOCS.items():
        path = _write_md_file(
            os.path.join(registry_dir, "behavioral"),
            filename,
            meta["title"],
            "behavioral",
            meta["tags"],
            "all",
            meta["content"],
        )
        md_files.append((path, "behavioral", meta["title"]))

    summary["docs_seeded"] = len(md_files)
    logger.info("Wrote %d markdown documents", summary["docs_seeded"])

    # ── 2b. Clean up stale seeder files ──────────────────────────────────
    # Remove .md files that were written by a *previous* seed but are no
    # longer in the current seed dictionaries.  Only delete files whose
    # frontmatter has source="seeder" — this is the seeder's unique
    # signature.  Files from `kb update` do not have this field and are
    # always preserved.
    _seed_filenames: dict[str, set[str]] = {
        "patterns": set(_PATTERN_DOCS.keys()),
        "adrs": set(_ADR_DOCS.keys()),
        "docs": set(_DOC_DOCS.keys()),
        "behavioral": set(_BEHAVIORAL_DOCS.keys()),
    }
    for subdir, expected in _seed_filenames.items():
        cat_dir = os.path.join(registry_dir, subdir)
        if not os.path.isdir(cat_dir):
            continue
        for fname in os.listdir(cat_dir):
            if fname.endswith(".md") and fname not in expected:
                stale_path = os.path.join(cat_dir, fname)
                try:
                    with open(stale_path, encoding="utf-8") as fh:
                        head = fh.read(500)
                    meta = _parse_frontmatter(head)
                    if meta.get("source") != "seeder":
                        continue  # not a seeder file — keep it
                    os.remove(stale_path)
                    logger.info("Removed stale seeder file: %s/%s", subdir, fname)
                except OSError as exc:
                    logger.debug("Failed to remove stale file %s: %s", stale_path, exc)

    # ── 3. Embed into SQLite vector store (optional) ──────────────────────
    # Embed ALL registry markdown files — both seeder-owned and files
    # from ``kb update`` — so that both sources coexist in the vector
    # store and appear in LLM prompt context.
    if embed:
        try:
            if api_client is None:
                raise ValueError("api_client required for embedding")

            all_md_files = collect_all_registry_md_files(
                exclude_paths={p for p, _, _ in md_files},
                registry_dir=registry_dir,
            )
            all_md_files = list(md_files) + all_md_files

            if len(all_md_files) > len(md_files):
                logger.info(
                    "Including %d additional file(s) from kb update in embedding",
                    len(all_md_files) - len(md_files),
                )

            embedded = _embed_md_files(all_md_files, project_root, api_client)
            summary["chunks_embedded"] = embedded
        except Exception as exc:
            logger.warning("Embedding skipped: %s", exc)
            summary["chunks_embedded"] = 0

    # ── 4. Write .seeded marker ─────────────────────────────────────────
    # Persistent marker so _global_kb_exists() can answer True without
    # touching the SQLite database.  This prevents spurious re-seeds
    # caused by transient DB lock errors during concurrent access.
    marker_path = os.path.join(core_dir, ".seeded")
    try:
        os.makedirs(os.path.dirname(marker_path), exist_ok=True)
        with open(marker_path, "w", encoding="utf-8") as fh:
            fh.write("seeded\n")
    except OSError as exc:
        logger.warning("Failed to write .seeded marker: %s", exc)

    return summary


_DIR_TO_CATEGORY = {
    "patterns": "pattern",
    "adrs": "adr",
    "docs": "doc",
    "behavioral": "behavioral",
}


def collect_all_registry_md_files(
    exclude_paths: Optional[set[str]] = None,
    registry_dir: Optional[str] = None,
) -> list[tuple[str, str, str]]:
    """Scan the registry for all ``.md`` files and return metadata tuples.

    This is used by both ``seed()`` and ``kb update`` (via CLI) to ensure
    ALL registry docs — from both sources — are embedded in the vector
    store so they coexist in LLM prompt context.

    Parameters
    ----------
    exclude_paths:
        Absolute file paths to skip (already collected by the caller).
    registry_dir:
        Override for the registry directory.  Defaults to the package
        ``_REGISTRY_DIR``.

    Returns
    -------
    list[tuple[str, str, str]]
        A list of ``(absolute_path, category, title)`` tuples suitable
        for passing to :func:`_embed_md_files`.
    """
    _reg_dir = registry_dir or _REGISTRY_DIR
    exclude = exclude_paths or set()
    md_files: list[tuple[str, str, str]] = []

    for subdir, category in _DIR_TO_CATEGORY.items():
        cat_dir = os.path.join(_reg_dir, subdir)
        if not os.path.isdir(cat_dir):
            continue
        for fname in os.listdir(cat_dir):
            if not fname.endswith(".md"):
                continue
            fpath = os.path.join(cat_dir, fname)
            if fpath in exclude:
                continue
            try:
                with open(fpath, encoding="utf-8") as fh:
                    head = fh.read(500)
                meta = _parse_frontmatter(head)
                title = meta.get("title", fname)
            except OSError:
                title = fname
            md_files.append((fpath, category, title))

    return md_files


def _embed_md_files(
    md_files: list[tuple[str, str, str]],
    project_root: str,
    api_client,
) -> int:
    """
    Embed markdown files into the `global_kb` collection.

    Reuses the embedding helpers from kb.local.embedder.
    Skips files whose content hasn't changed since last embedding
    (tracked via a content-hash marker in the vector store metadata).

    Returns the total number of chunks embedded.
    """
    import hashlib
    from ..local.embedder import _embed_batch, BATCH_SIZE, make_point_id
    from .store import _get_global_vector_store
    from ...config import Config

    cfg = Config.load()
    embed_model = cfg.EMBEDDING_MODEL or cfg.DEFAULT_MODEL

    # Create a store for the global_kb collection
    store = _get_global_vector_store()
    total_chunks = 0

    # Load existing content hashes to skip unchanged files
    existing_hashes: dict[str, str] = {}
    try:
        existing_hashes = store.get_metadata_map("file", "content_hash")
    except Exception:
        pass  # store may not support this yet — embed everything

    # Collect all chunks across files, then embed in one big batch
    all_chunk_data: list[tuple[str, str, str, str, str, list[str], str]] = []
    # Each entry: (rel_path, category, title, language, version, tags, chunk_text)

    for filepath, category, title in md_files:
        with open(filepath, encoding="utf-8") as fh:
            raw = fh.read()

        # Content hash for skip-if-unchanged
        content_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

        # Get relative path within registry (always use forward slashes
        # so that UUID5 point_ids are OS-independent and dedup works)
        rel_path = os.path.relpath(filepath, _GLOBAL_DIR).replace("\\", "/")

        # Skip if content unchanged
        if existing_hashes.get(rel_path) == content_hash:
            logger.debug("Skipping unchanged file: %s", rel_path)
            continue

        # Strip frontmatter
        body = raw
        if raw.startswith("---"):
            parts = raw.split("---", 2)
            if len(parts) >= 3:
                body = parts[2]

        # Parse frontmatter for metadata
        meta = _parse_frontmatter(raw)
        tags = [t.strip() for t in meta.get("tags", "").split(",") if t.strip()]
        language = meta.get("language", "all")
        version = meta.get("version", "1.0.0")

        chunks = _chunk_markdown(body, title)
        if not chunks:
            continue

        for chunk_text in chunks:
            all_chunk_data.append(
                (rel_path, category, title, language, version, tags,
                 content_hash, chunk_text)
            )

    if not all_chunk_data:
        logger.debug("No files need (re-)embedding — all content hashes match")
        return 0

    # Embed all chunks in batches
    all_texts = [cd[-1] for cd in all_chunk_data]
    for i in range(0, len(all_texts), BATCH_SIZE):
        batch_texts = all_texts[i : i + BATCH_SIZE]
        batch_data = all_chunk_data[i : i + BATCH_SIZE]
        try:
            vectors = _embed_batch(api_client, batch_texts, embed_model)
        except RuntimeError as exc:
            logger.warning("Embedding failed for batch starting at %d: %s", i, exc)
            continue

        points = []
        for j, ((rel_path, category, title, language, version, tags,
                  content_hash, chunk_text), vector) in enumerate(
                zip(batch_data, vectors)):
            point_id = str(uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"global:{rel_path}:{i + j}",
            ))
            payload = {
                "file": rel_path,
                "category": category,
                "title": title,
                "language": language,
                "tags": tags,
                "version": version,
                "source": "core",
                "content_hash": content_hash,
            }
            points.append((point_id, vector, payload))

        store.upsert(points)
        total_chunks += len(batch_texts)

    return total_chunks


def _parse_frontmatter(text: str) -> dict[str, str]:
    """Parse YAML frontmatter from markdown text.

    Handles both inline values (``tags: "a, b, c"``) and YAML list
    format::

        tags:
          - a
          - b
          - c

    List items are joined with ``", "`` so downstream code can treat
    them identically to inline comma-separated values.
    """
    if not text.startswith("---"):
        return {}
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}
    meta: dict[str, str] = {}
    current_key: str | None = None
    list_items: list[str] = []

    for line in parts[1].strip().split("\n"):
        stripped = line.strip()
        # YAML list item (e.g. "  - tailwindcss")
        if stripped.startswith("- ") and current_key is not None:
            list_items.append(stripped[2:].strip().strip('"').strip("'"))
            continue
        # Flush any accumulated list items into the previous key
        if list_items and current_key is not None:
            meta[current_key] = ", ".join(list_items)
            list_items = []
            current_key = None
        # Regular key: value line
        if ":" in line:
            key, _, val = line.partition(":")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if val:
                meta[key] = val
                current_key = None
            else:
                # Value is empty — next lines might be YAML list items
                current_key = key
                meta[key] = ""
    # Flush trailing list items
    if list_items and current_key is not None:
        meta[current_key] = ", ".join(list_items)

    return meta



