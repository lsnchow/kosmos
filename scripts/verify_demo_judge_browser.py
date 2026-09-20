"""One explicit real browser assessment, followed by a read-only reload check."""
import json
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from plumb.demo_judge import _assessment_from_report

output = ROOT / "data/private/semantic-judge" / (sys.argv[1] if len(sys.argv) > 1 else "browser-acceptance")
output.mkdir(exist_ok=False)
with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1440, "height": 1000})
    errors, posts = [], []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.on("request", lambda request: posts.append(request.url) if request.method == "POST" else None)
    page.goto("http://127.0.0.1:8787/console", wait_until="networkidle")
    assert not posts
    with page.expect_response(lambda response: response.request.method == "POST" and response.url.endswith("/api/demo/judgments"), timeout=30000) as submitted:
        page.get_by_role("button", name="Assess with our trained judge", exact=True).click()
    response = submitted.value
    record = response.json()
    (output / "admission.json").write_text(json.dumps(record, indent=2))
    assert response.status == 202, record
    request_id = record["id"]
    print("Admitted", request_id, flush=True)
    started = time.monotonic()
    while record["status"] not in {"completed", "abstained", "failed", "interrupted"}:
        page.wait_for_timeout(2000)
        record = page.request.get("http://127.0.0.1:8787/api/demo/judgments/" + request_id).json()
        if int(time.monotonic() - started) % 20 < 2:
            print(record["status"], round(time.monotonic() - started), "seconds", flush=True)
        if time.monotonic() - started > 630:
            raise RuntimeError("Timed out observing persisted request; do not resubmit")
    (output / "judgment.json").write_text(json.dumps(record, indent=2))
    print("Terminal", record["status"], flush=True)
    assert record["status"] in {"completed", "abstained"}, record.get("error")
    result = record["result"]
    expected = _assessment_from_report(result["report"])
    assert result["assessment"] == expected
    receipt = result["adapter_receipt"]
    assert receipt["adapter_enabled"] is True and receipt["verified_enabled_layer_count"] > 0
    assert receipt["active_adapter"] == "semantic_pilot_epoch_02"
    page.wait_for_timeout(2000)
    for metric in page.locator(".judge-metrics div").all():
        label = metric.locator("dt").inner_text()
        actual = metric.locator("dd").inner_text()
        field = {"Task progress": "progress", "Visual integrity": "visual_integrity", "Collision": "collision", "Completion": "completion"}[label]
        wanted = expected[field]
        if field == "progress": wanted = "Unable to assess" if wanted is None else f"{wanted} / 5"
        assert actual == str(wanted), (label, actual, wanted)
    page.get_by_text("Inference breakdown", exact=True).click()
    assert page.get_by_text(request_id, exact=True).is_visible()
    page.screenshot(path=str(output / "assessment.png"), full_page=True)
    before = len(posts)
    page.reload(wait_until="networkidle")
    page.get_by_text("Inference breakdown", exact=True).click()
    assert page.get_by_text(request_id, exact=True).is_visible()
    page.wait_for_timeout(2000)
    restored = page.request.get("http://127.0.0.1:8787/api/demo/judgments/" + request_id).json()
    assert restored["result"] == result
    assert before == len(posts) == 1, posts
    assert not errors, errors
    page.screenshot(path=str(output / "reload.png"), full_page=True)
    (output / "verification.json").write_text(json.dumps({"request_id": request_id, "status": record["status"], "browser_post_count": len(posts), "reload_resubmissions": 0, "display_matches_raw_report": True, "page_errors": errors, "adapter_receipt": receipt, "timing": result["timing"]}, indent=2))
    print("Verified", request_id, "one POST, same persisted result after reload", flush=True)
    browser.close()
