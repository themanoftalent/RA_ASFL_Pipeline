"""Check the official data landing pages without downloading ECG records."""
import argparse
import json
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DATASETS = {
    "MIT-BIH Arrhythmia Database": "https://physionet.org/content/mitdb/1.0.0/",
    "PTB-XL": "https://physionet.org/content/ptb-xl/1.0.3/",
}


def check(timeout=15):
    results = []
    for name, url in DATASETS.items():
        row = {"dataset": name, "url": url, "available": None}
        try:
            request = Request(url, method="HEAD", headers={"User-Agent": "RA-ASFL-availability-check/1.0"})
            with urlopen(request, timeout=timeout) as response:
                row["http_status"] = response.status
                row["available"] = response.status == 200
        except HTTPError as exc:
            row["http_status"] = exc.code
            row["available"] = False
        except (URLError, TimeoutError, OSError) as exc:
            row["status"] = "could_not_check"
            row["detail"] = type(exc).__name__
        results.append(row)
    return {"checked_at_utc": datetime.now(timezone.utc).isoformat(), "datasets": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=float, default=15)
    args = parser.parse_args()
    print(json.dumps(check(args.timeout), indent=2))
