"""
Scrape Terminal-Bench 2.0 leaderboard from tbench.ai.

Extracts: agent_name, model_name, accuracy, std_error, organization,
verification status for all entries.
"""

import json
import re
import requests
import pandas as pd
from pathlib import Path

LEADERBOARD_URL = "https://www.tbench.ai/leaderboard/terminal-bench/2.0"
_BENCHMARK_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = str(_BENCHMARK_DIR / "raw")


def main():
    print(f"Fetching leaderboard from {LEADERBOARD_URL}...")
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
    }
    resp = requests.get(LEADERBOARD_URL, headers=headers, timeout=30)
    resp.raise_for_status()
    html = resp.text
    print(f"Got {len(html)} bytes of HTML")

    # Save raw HTML for inspection
    with open(f"{OUTPUT_DIR}/leaderboard_raw.html", "w") as f:
        f.write(html)
    print("Saved raw HTML")

    # Try to extract Next.js JSON data from __NEXT_DATA__
    next_data_match = re.search(
        r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', html, re.DOTALL
    )
    if next_data_match:
        next_data = json.loads(next_data_match.group(1))
        with open(f"{OUTPUT_DIR}/leaderboard_nextdata.json", "w") as f:
            json.dump(next_data, f, indent=2)
        print("Extracted __NEXT_DATA__ JSON")

        # Try to find leaderboard entries in the JSON
        def find_leaderboard_data(obj, path=""):
            """Recursively search for arrays that look like leaderboard entries."""
            if isinstance(obj, list) and len(obj) > 5:
                # Check if items look like leaderboard entries
                if isinstance(obj[0], dict):
                    keys = set(obj[0].keys())
                    if any(k in keys for k in
                           ["accuracy", "score", "agent", "model",
                            "agent_name", "model_name", "p_hat"]):
                        print(f"  Found candidate at {path}: "
                              f"{len(obj)} items, keys={keys}")
                        return obj
            if isinstance(obj, dict):
                for k, v in obj.items():
                    result = find_leaderboard_data(v, f"{path}.{k}")
                    if result:
                        return result
            if isinstance(obj, list):
                for i, item in enumerate(obj):
                    result = find_leaderboard_data(item, f"{path}[{i}]")
                    if result:
                        return result
            return None

        entries = find_leaderboard_data(next_data)
        if entries:
            df = pd.DataFrame(entries)
            df.to_csv(f"{OUTPUT_DIR}/leaderboard_results.csv", index=False)
            print(f"\nExtracted {len(df)} leaderboard entries")
            print(f"Columns: {list(df.columns)}")
            print(f"\nFirst 5 entries:")
            print(df.head().to_string())
            return

    # Fallback: try parsing from HTML tables or divs
    print("\nAttempting HTML table extraction...")
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")

    # Look for tables
    tables = soup.find_all("table")
    if tables:
        print(f"Found {len(tables)} HTML tables")
        for i, table in enumerate(tables):
            rows = table.find_all("tr")
            print(f"  Table {i}: {len(rows)} rows")

    # Look for structured data in script tags
    scripts = soup.find_all("script")
    for script in scripts:
        text = script.string or ""
        if "accuracy" in text or "leaderboard" in text or "p_hat" in text:
            print(f"\nFound script with relevant data ({len(text)} chars)")
            # Save for manual inspection
            with open(f"{OUTPUT_DIR}/leaderboard_script_data.txt", "w") as f:
                f.write(text[:50000])

    # Try looking for JSON-LD or other structured data
    json_ld = soup.find_all("script", type="application/ld+json")
    if json_ld:
        for j in json_ld:
            data = json.loads(j.string)
            print(f"Found JSON-LD: {json.dumps(data, indent=2)[:500]}")

    # Also look at the RSC (React Server Components) payload
    rsc_scripts = [s for s in scripts if s.get("src", "").endswith(".js")]
    print(f"\nFound {len(rsc_scripts)} JS script tags")

    # Try the API endpoint directly
    print("\nTrying potential API endpoints...")
    api_urls = [
        "https://www.tbench.ai/api/leaderboard",
        "https://www.tbench.ai/api/leaderboard/terminal-bench/2.0",
        "https://www.tbench.ai/api/results",
    ]
    for url in api_urls:
        try:
            r = requests.get(url, headers=headers, timeout=10)
            if r.status_code == 200:
                print(f"  {url}: {r.status_code} ({len(r.text)} bytes)")
                data = r.json()
                with open(f"{OUTPUT_DIR}/leaderboard_api.json", "w") as f:
                    json.dump(data, f, indent=2)
                print(f"  Saved API response")
            else:
                print(f"  {url}: {r.status_code}")
        except Exception as e:
            print(f"  {url}: {e}")

    print("\n=== Manual inspection may be needed ===")
    print("Check leaderboard_raw.html and leaderboard_nextdata.json")


if __name__ == "__main__":
    main()
