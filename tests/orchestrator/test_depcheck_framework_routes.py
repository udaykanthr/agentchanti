"""A file a framework loads by location is never an orphaned export.

Measured 2026-09-22 (glm-5.3-flash, Next.js home page): `my-app/app/page.tsx
exports [Home] but no other file imports it` fired after two waves and cost
two LLM fix calls, 11.8k output tokens, for a gap that cannot exist — the
App Router loads page.tsx from its path, and the only available "fix"
(importing it into layout.tsx) renders it outside the router.
"""
import json

from agentchanti.orchestrator.dependency_check import (
    DependencySnapshot, FileDeps, find_gaps, framework_route_reason)

NEXT_PKG = json.dumps({"dependencies": {"next": "16.0.0", "react": "19"}})
VITE_PKG = json.dumps({"devDependencies": {"vite": "7"}})


def _gaps(files: dict[str, str], new: str, exports=("Home",)):
    deps = {p: FileDeps(file_path=p) for p in files}
    deps[new] = FileDeps(file_path=new, exports=list(exports))
    deps["my-app/app/layout.tsx"] = FileDeps(
        file_path="my-app/app/layout.tsx", exports=["RootLayout"])
    before = DependencySnapshot(file_deps={})
    after = DependencySnapshot(file_deps=deps)
    return [g for g in find_gaps(before, after, [new], "step", files)
            if g.gap_type == "orphaned_export"]


class TestTheMeasuredCase:

    def test_next_app_page_is_not_orphaned(self):
        files = {"my-app/package.json": NEXT_PKG,
                 "my-app/app/page.tsx": "export default function Home(){}",
                 "my-app/app/layout.tsx": "export default function L(){}"}
        assert _gaps(files, "my-app/app/page.tsx") == []

    def test_the_reason_names_the_framework(self):
        files = {"my-app/package.json": NEXT_PKG}
        assert framework_route_reason("my-app/app/page.tsx", files) \
            == "Next.js App Router"


class TestWhatIsStillJudged:

    def test_a_component_inside_app_is_still_checked(self):
        """app/components/Hero.tsx is not a route — unused, it is a gap."""
        files = {"my-app/package.json": NEXT_PKG,
                 "my-app/app/components/Hero.tsx": "export function Hero(){}"}
        assert framework_route_reason(
            "my-app/app/components/Hero.tsx", files) is None
        assert len(_gaps(files, "my-app/app/components/Hero.tsx",
                         exports=("Hero",))) == 1

    def test_page_tsx_without_next_is_an_ordinary_module(self):
        files = {"web/package.json": VITE_PKG}
        assert framework_route_reason("web/app/page.tsx", files) is None

    def test_no_package_json_means_no_claim(self):
        assert framework_route_reason("app/page.tsx", {}) is None

    def test_the_nearest_manifest_decides_in_a_two_app_repo(self):
        files = {"site/package.json": NEXT_PKG, "admin/package.json": VITE_PKG}
        assert framework_route_reason("site/app/page.tsx", files)
        assert framework_route_reason("admin/app/page.tsx", files) is None


class TestOtherConventions:

    def test_next_src_app_pages_router_and_middleware(self):
        files = {"package.json": NEXT_PKG}
        assert framework_route_reason("src/app/dashboard/page.tsx", files)
        assert framework_route_reason("app/api/users/route.ts", files)
        assert framework_route_reason("pages/about.tsx", files)
        assert framework_route_reason("middleware.ts", files)
        assert framework_route_reason("lib/middleware.ts", files) is None

    def test_sveltekit_plus_files(self):
        files = {"package.json": json.dumps(
            {"devDependencies": {"@sveltejs/kit": "2"}})}
        assert framework_route_reason("src/routes/+page.svelte", files)
        assert framework_route_reason("src/routes/+server.ts", files)
        assert framework_route_reason("src/lib/Button.svelte", files) is None

    def test_vite_entry_loaded_by_index_html(self):
        files = {"package.json": VITE_PKG,
                 "index.html": '<script type="module" src="/src/main.tsx">'
                               "</script>"}
        assert framework_route_reason("src/main.tsx", files) \
            == "index.html <script>"
        assert framework_route_reason("src/other.tsx", files) is None

    def test_a_dotfile_prefix_is_not_stripped(self):
        """The lstrip('./') trap: `.app/page.tsx` is not `app/page.tsx`."""
        files = {"package.json": NEXT_PKG}
        assert framework_route_reason(".app/page.tsx", files) is None
