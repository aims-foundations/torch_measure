"""
01_download_raw.py — OECD AI Incidents Monitor (AIM) placeholder.

Source: https://oecd.ai/en/incidents

NOTE: The OECD AIM does not provide a direct bulk download API or data export.
The incident data is accessible through a web interface only.
This script downloads the methodology documentation as a reference and creates
a placeholder noting that manual download is required.
"""

import urllib.request
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "raw"

PLACEHOLDER_TEXT = """\
OECD AI Incidents Monitor (AIM) — Manual Download Required
============================================================

The OECD AIM database does not provide a public bulk download API.

Data source: https://oecd.ai/en/incidents

Options for obtaining data:
1. Manual browsing and export from the OECD AIM web interface.
2. Contact OECD for research access to bulk data.
3. Use web scraping (check OECD terms of service first).
4. Check if OECD publishes periodic CSV/JSON dumps.

Once you have the data, place the files in this raw/ directory.

Reference:
- OECD AI Policy Observatory: https://oecd.ai/en/
- AIM Methodology: https://oecd.ai/en/dashboards/ai-incidents
"""


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("[INFO] OECD AI Incidents Monitor (AIM) — data download setup")

    # Create placeholder readme
    readme_file = RAW_DIR / "MANUAL_DOWNLOAD_NEEDED.txt"
    if readme_file.exists():
        print("[SKIP] Placeholder readme already exists.")
    else:
        readme_file.write_text(PLACEHOLDER_TEXT)
        print(f"[INFO] Created placeholder: {readme_file}")

    # Download methodology page as reference
    methodology_file = RAW_DIR / "oecd_aim_methodology.html"
    if methodology_file.exists():
        print("[SKIP] Methodology page already downloaded.")
    else:
        print("[INFO] Downloading OECD AIM methodology page as reference...")
        try:
            req = urllib.request.Request(
                "https://oecd.ai/en/incidents",
                headers={"User-Agent": "Mozilla/5.0 (compatible; research-download)"},
            )
            with urllib.request.urlopen(req) as resp:
                methodology_file.write_bytes(resp.read())
            print("[INFO] Saved methodology reference page.")
        except Exception as e:
            print(f"[WARN] Could not download methodology page: {e}")
            print("[INFO] Visit https://oecd.ai/en/incidents manually.")

    print("\n[INFO] OECD AIM setup complete.")
    print("[WARN] Bulk data must be obtained manually from https://oecd.ai/en/incidents")


if __name__ == "__main__":
    main()
