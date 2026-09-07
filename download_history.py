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

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_file = output_file.with_suffix(
        f"{output_file.suffix}.tmp"
    )

    with temporary_file.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)
        writer.writerows(rows)

    # Replace only after a complete CSV has been written so an interrupted
    # refresh cannot destroy the last usable current-day snapshot.
    temporary_file.replace(output_file)

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

        # Historical sheets are immutable. The current day's sheet is not: its
        # prices change throughout the day, so refresh it on every run.
        if output_file.exists() and current_date != today:
            print(
                f"Already downloaded: "
                f"{output_file.name}"
            )
            continue

        if output_file.exists():
            print(
                f"Refreshing current day: "
                f"{output_file.name}"
            )

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
        f"Downloaded or refreshed {downloaded} sheets."
    )


if __name__ == "__main__":
    main()
