"""Browser smoke test against a running local app; synthetic records only.
Install dev requirements and `python -m playwright install chromium` first.
Optional: --channel chrome to use locally installed Chrome.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from playwright.sync_api import expect, sync_playwright

parser = argparse.ArgumentParser()
parser.add_argument("--url", default="http://127.0.0.1:8501")
parser.add_argument("--channel", default=None)
parser.add_argument("--output-dir", type=Path, default=Path("outputs/part3"))
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
out = args.output_dir
out.mkdir(parents=True, exist_ok=True)
checks = []
with sync_playwright() as pw:
    browser = pw.chromium.launch(headless=True, channel=args.channel)
    page = browser.new_page(
        viewport={"width": 1440, "height": 1600}, device_scale_factor=1
    )
    page.goto(args.url)
    expect(page.get_by_role("heading", name="Student Review Studio")).to_be_visible(
        timeout=30000
    )
    page.get_by_role("button", name="Estimate probability", exact=True).click()
    expect(page.get_by_text("12.7%", exact=True)).to_be_visible()
    page.screenshot(path=str(out / "individual.png"), full_page=True)
    checks.append("individual prediction 12.7%")
    page.get_by_text("Capacity-based review", exact=True).click()
    page.locator("input[type=file]").set_input_files(
        str(root / "examples/synthetic_students.csv")
    )
    expect(page.get_by_text("Review list size", exact=True)).to_be_visible(
        timeout=30000
    )

    def download(expected, name):
        metric = page.locator("[data-testid=stMetric]").filter(
            has=page.get_by_text("Review list size", exact=True)
        )
        expect(metric.locator("[data-testid=stMetricValue]")).to_have_text(
            str(expected)
        )
        with page.expect_download() as pending:
            page.get_by_role(
                "button", name="Download review results", exact=True
            ).click()
        path = out / name
        pending.value.save_as(str(path))
        df = pd.read_csv(path)
        assert len(df) == 20 and df.selected_for_review.eq(True).sum() == expected
        return df

    download(2, "browser_download_10.csv")
    checks.append("CSV upload and actual download: 20 rows, 2 selected at 10%")
    page.get_by_role("combobox").click()
    page.get_by_role("option", name="20%", exact=True).click()
    download(4, "browser_download_20.csv")
    checks.append("capacity 20%: 4 selected")
    page.screenshot(path=str(out / "capacity.png"), full_page=True)
    page.get_by_text("Number of students", exact=True).click()
    page.get_by_role("spinbutton", name="Number to review").fill("3")
    page.get_by_role("spinbutton", name="Number to review").press("Enter")
    download(3, "browser_download_count.csv")
    checks.append("count capacity: 3 selected")
    bad = pd.read_csv(root / "examples/synthetic_students.csv")
    bad.loc[0, "Age at enrollment"] = -1
    page.locator("input[type=file]").set_input_files(
        {
            "name": "synthetic_invalid.csv",
            "mimeType": "text/csv",
            "buffer": bad.to_csv(index=False).encode(),
        }
    )
    expect(page.get_by_text("1 records rejected.", exact=False)).to_be_visible()
    expect(
        page.get_by_role("button", name="Download review results", exact=True)
    ).to_have_count(0)
    page.get_by_text(
        "I acknowledge the exclusions; calculate capacity using only the 19 valid records",
        exact=True,
    ).click()
    expect(
        page.get_by_role("checkbox", name="I acknowledge the exclusions;", exact=False)
    ).to_be_checked()
    page.get_by_role("spinbutton", name="Number to review").fill("3")
    page.get_by_role("spinbutton", name="Number to review").press("Enter")
    df = download(3, "browser_download_partial.csv")
    assert df.status.eq("rejected").sum() == 1
    page.screenshot(path=str(out / "validation.png"), full_page=True)
    checks.append(
        "invalid upload: 1 rejection, explicit acknowledgment, 20-row export with missing rejected probability"
    )
    assert not page.locator('[data-testid="stException"]').count()
    browser.close()
(out / "browser_smoke.json").write_text(
    json.dumps(
        {
            "status": "passed",
            "browser": "Chromium via Playwright"
            + (" channel " + args.channel if args.channel else ""),
            "checks": checks,
        },
        indent=2,
    )
)
print("\n".join(checks))
