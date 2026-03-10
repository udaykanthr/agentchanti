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
              "exact string matching fails, (3) the element uses a different role/tag than expected.",
        fix_template="CRITICAL FIX STEPS:\n"
                     "1. READ THE ACTUAL DOM OUTPUT shown in the error message — it shows exactly "
                     "what the component renders.\n"
                     "2. Match your query to ACTUAL rendered text, not assumed text.\n"
                     "3. For text split across elements, use a function matcher:\n"
                     "   screen.getByText((content, element) => element?.textContent === 'expected full text')\n"
                     "4. Prefer getByRole() over getByText() for more robust queries:\n"
                     "   screen.getByRole('button', { name: /submit/i })\n"
                     "   screen.getByRole('heading', { name: /welcome/i })\n"
                     "5. NEVER hardcode assumed text content — always derive expected values "
                     "from the actual source component props or rendered DOM.\n"
                     "6. Use { exact: false } for partial text matching: screen.getByText('partial', { exact: false })",
        severity="error",
        tags="testing-library,react,getByText,getByRole,element,not,found,query,dom,javascript,typescript",
    ),
    ErrorFix(
        error_type="TestingLibraryMultipleElementsFound",
        language="javascript",
        pattern=r"TestingLibraryElementError:\s*Found multiple elements with the role",
        cause="The test uses getByRole/getByText which expects exactly ONE matching element, "
              "but the component renders multiple elements with the same role and accessible name. "
              "Common causes: (1) a navigation bar and page content both have a 'Home' link, "
              "(2) header/footer duplicate links, (3) nested components render the same elements.",
        fix_template="FIX STEPS:\n"
                     "1. Use getAllByRole() instead of getByRole() when multiple matches are expected, "
                     "then assert on the array:\n"
                     "   const links = screen.getAllByRole('link', { name: /home/i })\n"
                     "   expect(links).toHaveLength(2)\n"
                     "   expect(links[0]).toBeInTheDocument()\n"
                     "2. Or scope the query to a specific container using within():\n"
                     "   import { within } from '@testing-library/react'\n"
                     "   const nav = screen.getByRole('navigation')\n"
                     "   within(nav).getByRole('link', { name: /home/i })\n"
                     "3. Use a more specific name pattern to narrow matches:\n"
                     "   screen.getByRole('link', { name: /^home$/i })  // exact match\n"
                     "4. READ the actual component source to understand how many instances exist "
                     "and which container to scope your query to.",
        severity="error",
        tags="testing-library,react,multiple,elements,getByRole,getAllByRole,within,query,javascript,typescript",
    ),
    ErrorFix(
        error_type="ReactRouterContextMissing",
        language="javascript",
        pattern=r"(useLocation|useNavigate|useParams|useMatch|useHref)\(\) may be used only in the context of a <Router> component",
        cause="The component uses React Router hooks (useLocation, useNavigate, useParams, etc.) "
              "or components (Link, NavLink, Outlet) but the test renders it without a Router wrapper. "
              "React Router hooks MUST be called inside a Router context.",
        fix_template="FIX: Wrap the component in <MemoryRouter> in your test:\n\n"
                     "import { MemoryRouter } from 'react-router-dom'\n\n"
                     "render(\n"
                     "  <MemoryRouter initialEntries={['/current-path']}>\n"
                     "    <YourComponent />\n"
                     "  </MemoryRouter>\n"
                     ")\n\n"
                     "KEY RULES:\n"
                     "1. ALWAYS use MemoryRouter (not BrowserRouter) in tests — it doesn't need a real DOM.\n"
                     "2. Set initialEntries to control the starting route:\n"
                     "   <MemoryRouter initialEntries={['/dashboard']}>\n"
                     "3. If testing components that use <Outlet>, provide matching <Routes>:\n"
                     "   <MemoryRouter initialEntries={['/dashboard']}>\n"
                     "     <Routes>\n"
                     "       <Route path='/dashboard' element={<Dashboard />} />\n"
                     "     </Routes>\n"
                     "   </MemoryRouter>\n"
                     "4. If mocking useNavigate, use vi.mock('react-router-dom') BEFORE the render.\n"
                     "5. Check ALL child components too — if ANY nested component uses Link/NavLink/useNavigate, "
                     "the entire tree needs the Router wrapper.",
        severity="error",
        tags="react-router,useLocation,useNavigate,useParams,MemoryRouter,Router,context,testing,react,javascript,typescript",
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
                     "Common missing test packages:\n"
                     "  npm install --save-dev @testing-library/user-event\n"
                     "  npm install --save-dev @testing-library/jest-dom\n"
                     "  npm install --save-dev @testing-library/react\n"
                     "  npm install --save-dev react-router-dom\n\n"
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
    summary = {
        "errors_seeded": 0,
        "content_fixes_seeded": 0,
        "docs_seeded": 0,
        "chunks_embedded": 0,
    }

    # ── 1a. Seed errors.db ──────────────────────────────────────────────
    db_path = _errors_db_path()
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
            os.path.join(_REGISTRY_DIR, "patterns"),
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
            os.path.join(_REGISTRY_DIR, "adrs"),
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
            os.path.join(_REGISTRY_DIR, "docs"),
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
            os.path.join(_REGISTRY_DIR, "behavioral"),
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
        cat_dir = os.path.join(_REGISTRY_DIR, subdir)
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

            all_md_files = collect_all_registry_md_files(exclude_paths={p for p, _, _ in md_files})
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
    marker_path = os.path.join(_CORE_DIR, ".seeded")
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
) -> list[tuple[str, str, str]]:
    """Scan the registry for all ``.md`` files and return metadata tuples.

    This is used by both ``seed()`` and ``kb update`` (via CLI) to ensure
    ALL registry docs — from both sources — are embedded in the vector
    store so they coexist in LLM prompt context.

    Parameters
    ----------
    exclude_paths:
        Absolute file paths to skip (already collected by the caller).

    Returns
    -------
    list[tuple[str, str, str]]
        A list of ``(absolute_path, category, title)`` tuples suitable
        for passing to :func:`_embed_md_files`.
    """
    exclude = exclude_paths or set()
    md_files: list[tuple[str, str, str]] = []

    for subdir, category in _DIR_TO_CATEGORY.items():
        cat_dir = os.path.join(_REGISTRY_DIR, subdir)
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

    Returns the total number of chunks embedded.
    """
    from ..local.embedder import _embed_batch, BATCH_SIZE, make_point_id
    from .store import _get_global_vector_store
    from ...config import Config

    cfg = Config.load()
    embed_model = cfg.EMBEDDING_MODEL or cfg.DEFAULT_MODEL

    # Create a store for the global_kb collection
    store = _get_global_vector_store()
    total_chunks = 0

    for filepath, category, title in md_files:
        with open(filepath, encoding="utf-8") as fh:
            raw = fh.read()

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

        # Get relative path within registry (always use forward slashes
        # so that UUID5 point_ids are OS-independent and dedup works)
        rel_path = os.path.relpath(filepath, _GLOBAL_DIR).replace("\\", "/")

        chunks = _chunk_markdown(body, title)
        if not chunks:
            continue

        # Embed in batches
        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i : i + BATCH_SIZE]
            try:
                vectors = _embed_batch(api_client, batch, embed_model)
            except RuntimeError as exc:
                logger.warning("Embedding failed for %s: %s", filepath, exc)
                continue

            points = []
            for j, (chunk_text, vector) in enumerate(zip(batch, vectors)):
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
                }
                points.append((point_id, vector, payload))

            store.upsert(points)
            total_chunks += len(batch)

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



