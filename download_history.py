import csv
import io
from datetime import date, timedelta
from pathlib import Path

import requests


# ============================================================
# Configuration
# ============================================================

SPREADSHEET_ID = "1lzRgDYbG-HFUneIvm76XMF6vtIO0guOox2QhTGLHV1U"

OUTPUT_DIR = Path("price_history")

SHEET_PREFIX = "Economy_History_"

# How far back to look.
DAYS_TO_CHECK = 365


# ============================================================
# Download one sheet
# ============================================================

def download_sheet(sheet_name, output_file):
    """
    Download a Google Sheets tab as a CSV file.

    Returns True if the sheet exists and was downloaded.
    Returns False if the sheet doesn't exist.
    """

    url = (
        f"https://docs.google.com/spreadsheets/d/"
        f"{SPREADSHEET_ID}/gviz/tq"
        f"?tqx=out:csv"
        f"&sheet={sheet_name}"
    )

    print(f"Checking {sheet_name}...")

    response = requests.get(
        url,
        timeout=30,
    )

    response.raise_for_status()

    # A nonexistent sheet usually causes Google to return
    # an error rather than normal CSV data.
    text = response.text

    if not text.strip():
        return False

    # Make sure this actually looks like CSV data.
    if "<html" in text.lower():
        return False

    rows = csv.reader(
        io.StringIO(text)
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_file.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)
        writer.writerows(rows)

    return True


# ============================================================
# Main
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    today = date.today()

    downloaded = 0

    print(
        f"Checking the last {DAYS_TO_CHECK} days...\n"
    )

    for days_ago in range(DAYS_TO_CHECK):

        current_date = today - timedelta(
            days=days_ago
        )

        date_string = current_date.isoformat()

        sheet_name = (
            f"{SHEET_PREFIX}{date_string}"
        )

        output_file = (
            OUTPUT_DIR /
            f"{date_string}.csv"
        )

        # Don't download files we already have.
        if output_file.exists():
            print(
                f"Already downloaded: "
                f"{output_file.name}"
            )
            continue

        try:

            success = download_sheet(
                sheet_name,
                output_file,
            )

            if success:
                print(
                    f"  -> Saved {output_file}"
                )

                downloaded += 1

            else:
                print(
                    "  -> Sheet not found"
                )

        except requests.RequestException as e:

            print(
                f"  -> Error: {e}"
            )

    print()
    print(
        f"Downloaded {downloaded} new sheets."
    )


if __name__ == "__main__":
    main()