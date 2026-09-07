import tempfile
import sys
import types
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

try:
    import requests  # noqa: F401
except ImportError:
    requests_stub = types.ModuleType("requests")
    requests_stub.RequestException = RuntimeError
    sys.modules["requests"] = requests_stub

import download_history


class DownloadHistoryTests(unittest.TestCase):
    def test_existing_current_day_is_refreshed(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            today_file = output_dir / f"{date.today().isoformat()}.csv"
            today_file.write_text("old", encoding="utf-8")

            with patch.object(download_history, "OUTPUT_DIR", output_dir), patch.object(
                download_history, "DAYS_TO_CHECK", 1
            ), patch.object(download_history, "download_sheet", return_value=True) as downloader:
                download_history.main()

            downloader.assert_called_once()


if __name__ == "__main__":
    unittest.main()
