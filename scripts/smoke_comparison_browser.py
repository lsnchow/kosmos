"""Real-browser localhost smoke; no mock routes or automatic GPU submissions.

Default checks the honest blocked/available state, media, mobile overflow, and
absence of inference POSTs on entry. --require-comparison demands a promoted
real 12-cell set. --steer explicitly sends one ArrowRight command to its first
cell; only use it with an existing approved server-side budget reservation.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

from playwright.sync_api import sync_playwright


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8787")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--require-comparison", action="store_true")
    parser.add_argument("--steer", action="store_true")
    args = parser.parse_args()
    parsed = urlsplit(args.base_url)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"}:
        parser.error("This smoke is scoped to the localhost application")
    if args.steer and not args.require_comparison:
        parser.error("--steer requires --require-comparison")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {"schema": "kosmos-browser-smoke-v1", "started_at": datetime.now(timezone.utc).isoformat(),
              "mocked": False, "steering_requested": args.steer, "checks": [], "page_errors": []}
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 1000}, reduced_motion="reduce")
        context.tracing.start(screenshots=True, snapshots=True)
        page = context.new_page()
        posts = []
        page.on("request", lambda request: posts.append(request.url) if request.method == "POST" else None)
        page.on("pageerror", lambda error: report["page_errors"].append(str(error)))
        try:
            readiness = context.request.get(args.base_url + "/api/comparisons/readiness")
            assert readiness.ok and "application/json" in readiness.headers.get("content-type", ""), "New comparison API is not running"
            report["readiness"] = readiness.json()
            page.goto(args.base_url + "/console", wait_until="networkidle")
            wall = page.locator("#matched-control-wall")
            wall.wait_for()
            page.get_by_text("Loading the promoted comparison record…").wait_for(state="hidden")
            table = wall.get_by_role("table", name="Three policies across four matched world seeds")
            if table.count():
                assert table.locator("tbody tr").count() == 4
                assert table.locator("tbody td article").count() == 12
                assert wall.get_by_text("Outcome: not scored", exact=True).count() == 12
                report["checks"].append("real promoted 3x4 set rendered")
            else:
                assert not args.require_comparison, "A complete promoted comparison is required but absent"
                assert wall.get_by_text("No promoted 12-cell set is available.", exact=False).count() == 1
                report["checks"].append("missing comparison is explicitly blocked, no substituted tiles")
            assert not posts, "Opening console unexpectedly submitted work"
            page.screenshot(path=str(args.output_dir / "desktop.png"), full_page=True)
            for width in (390, 768):
                page.set_viewport_size({"width": width, "height": 844})
                assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth"), f"Page overflows at {width}px"
                report["checks"].append(f"{width}px viewport has no page overflow")
            page.screenshot(path=str(args.output_dir / "mobile.png"), full_page=True)
            assert not posts, "Viewport/reduced-motion changes unexpectedly submitted work"
            report["checks"].append("entry and responsive inspection made zero POSTs")
            if args.steer:
                page.set_viewport_size({"width": 1440, "height": 1000})
                control = wall.get_by_role("button", name="Take control", exact=False).first
                assert control.is_enabled(), "Manual control is unavailable"
                control.click()
                dialog = page.get_by_role("dialog")
                dialog.wait_for()
                surface = dialog.locator("[tabindex='0']").first
                surface.focus()
                before = len([url for url in posts if url.endswith("/commands")])
                started = time.monotonic()
                page.keyboard.down("ArrowRight")
                page.keyboard.down("ArrowRight")
                page.keyboard.up("ArrowRight")
                # Repeated keydown is deliberately one command. No automatic retry.
                page.wait_for_function("() => !document.querySelector('[aria-label=\"Manual branch frame\"][aria-busy=\"true\"]')", timeout=600000)
                after = len([url for url in posts if url.endswith("/commands")])
                assert after - before == 1, "One held key must produce exactly one command"
                assert dialog.get_by_text("16 committed post-conditioning frames", exact=True).count() == 1
                report["manual_client_seconds"] = time.monotonic() - started
                report["checks"].append("one real manual segment committed after one ArrowRight command")
                page.screenshot(path=str(args.output_dir / "manual.png"), full_page=True)
                page.keyboard.press("Escape")
            assert not report["page_errors"], "Browser emitted JavaScript errors"
            report["status"] = "passed"
        except Exception as error:
            report["status"] = "failed"
            report["error"] = str(error)
        finally:
            report["post_requests"] = posts
            context.tracing.stop(path=str(args.output_dir / "trace.zip"))
            browser.close()
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "checks": report["checks"], "error": report.get("error")}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
