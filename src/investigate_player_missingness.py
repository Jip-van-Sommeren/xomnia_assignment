"""Locate blank player histories without changing or filling source data.

Run from the project root with:
    python3 src/investigate_player_missingness.py

Uses the standard library and bounded memory: summaries per app and timestamp.
"""

import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports" / "step2"


def profile(path, cadence_seconds, metadata, previous):
    apps = {}
    timestamps = Counter()
    missing_timestamps = Counter()
    order_errors = 0
    samples = []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, strict=True)
        if next(reader) != ["app_id", "Time", "Playercount"]:
            raise ValueError(f"Unexpected header: {path}")
        for record, row in enumerate(reader, start=2):
            if len(row) != 3:
                raise ValueError(f"Unexpected row width: {path}, record {record}")
            identifier, timestamp, value = (cell.strip() for cell in row)
            int(identifier)  # Fail on an unexpected application ID.
            app = apps.setdefault(identifier, {
                "rows": 0, "blank": 0, "first": timestamp, "last": timestamp,
            })
            if app["rows"] and timestamp <= app["last"]:
                order_errors += 1
            app["rows"] += 1
            app["last"] = timestamp
            timestamps[timestamp] += 1
            if not value:
                app["blank"] += 1
                missing_timestamps[timestamp] += 1
                if len(samples) < 3:
                    samples.append({"csv_record": record, "fields": row})

    times = sorted(datetime.fromisoformat(t) for t in timestamps)
    intervals = Counter(
        int((right - left).total_seconds())
        for left, right in zip(times, times[1:])
    )
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    total = sum(app["rows"] for app in apps.values())
    missing = sum(app["blank"] for app in apps.values())
    previous_source = next(s for s in previous["sources"]
                           if Path(s["file"]).name == path.name)
    matches_audit = (
        digest == previous_source["sha256"]
        and total == previous_source["rows_scanned"]
        and missing == previous_source["numeric_columns"]["Playercount"]["missing"]
    )
    # Strictly increasing timestamps per app imply distinct keys. If each app
    # has as many keys as the global timestamp set, all share the entire grid.
    complete_grid = (
        order_errors == 0
        and all(app["rows"] == len(times) for app in apps.values())
        and all(count == len(apps) for count in timestamps.values())
        and set(intervals) == {cadence_seconds}
    )
    result = {
        "source": path.name,
        "sha256": digest,
        "matches_original_audit_bytes_and_counts": matches_audit,
        "rows": total,
        "applications": len(apps),
        "timestamps": len(times),
        "start": str(times[0]),
        "end": str(times[-1]),
        "complete_application_timestamp_grid": complete_grid,
        "non_increasing_timestamp_pairs": order_errors,
        "rows_per_application_histogram": dict(Counter(a["rows"] for a in apps.values())),
        "blank_values": missing,
        "blank_percent": 100 * missing / total,
        "applications_with_no_blanks": sum(a["blank"] == 0 for a in apps.values()),
        "applications_partially_blank": sum(0 < a["blank"] < a["rows"] for a in apps.values()),
        "applications_entirely_blank": [
            {"app_id": int(identifier), **app, "metadata": metadata[identifier]}
            for identifier, app in apps.items() if app["blank"] == app["rows"]
        ],
        "blank_values_per_timestamp_histogram": dict(Counter(
            missing_timestamps[t] for t in timestamps
        )),
        "first_blank_records": samples,
    }
    coverage = [
        {"source": path.name, "app_id": identifier,
         "name": metadata[identifier]["name"], "type": metadata[identifier]["type"],
         **app, "nonblank": app["rows"] - app["blank"]}
        for identifier, app in sorted(apps.items(), key=lambda item: int(item[0]))
    ]
    return result, coverage


def main():
    with (ROOT / "data" / "applicationInformation.csv").open(
        encoding="cp1252", newline=""
    ) as stream:
        metadata = {row["appid"]: row for row in csv.DictReader(stream)}
    previous = json.loads((OUTPUT / "report.json").read_text())
    results, coverage = [], []
    for name, cadence in (("Playercount_top1000.csv", 3600),
                          ("Playercount_bottom1000.csv", 86400)):
        result, rows = profile(ROOT / "data" / name, cadence, metadata, previous)
        results.append(result)
        coverage.extend(rows)
        print(json.dumps(result, indent=2), flush=True)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Both complete source files; blank means empty/whitespace field",
        "metadata_encoding": "cp1252, provisional as in the original source audit",
        "sources": results,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "player_missingness.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    with (OUTPUT / "player_missingness_by_application.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(coverage[0]))
        writer.writeheader()
        writer.writerows(coverage)


if __name__ == "__main__":
    main()
