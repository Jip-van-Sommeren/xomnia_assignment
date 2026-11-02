"""Profile the ten assessment CSVs without modifying them or filling missing data.

Histories are processed in chunks. A temporary, disk-backed SQLite index checks
business-key uniqueness across all scanned rows, including chunk boundaries.
Run from the project root: python src/analyze_sources.py --help
"""

from __future__ import annotations

import argparse
import codecs
import csv
import hashlib
import json
import re
import sqlite3
import tempfile
from collections import Counter
from contextlib import closing
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INFORMATION = "applicationInformation.csv"
HEADERS = {
    INFORMATION: ["appid", "type", "name", "releasedate", "freetoplay"],
    "Playercount_top1000.csv": ["app_id", "Time", "Playercount"],
    "Playercount_bottom1000.csv": ["app_id", "Time", "Playercount"],
    "Priceshistory.csv": [
        "app_id",
        "Date",
        "Initialprice",
        "Finalprice",
        "Discount",
    ],
}
RELATIONSHIPS = {
    "applicationDevelopers.csv": "developer",
    "applicationPublishers.csv": "publisher",
    "applicationSupportedlanguages.csv": "language",
    "applicationPackages.csv": "package_id",
    "applicationGenres.csv": "genre",
    "applicationTags.csv": "tag",
}


def discover_sources(data_dir: Path) -> dict[str, Path]:
    candidates = [
        p
        for directory in (data_dir, data_dir / "raw")
        if directory.is_dir()
        for p in directory.iterdir()
        if p.is_file()
    ]
    sources = {}
    for name in [*HEADERS, *RELATIONSHIPS]:
        aliases = {name.lower()}
        if name == "Priceshistory.csv":
            aliases.add("pricehistory.csv")
        matches = [p for p in candidates if p.name.lower() in aliases]
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one {name} in {data_dir} or its raw/ folder; found {matches}"
            )
        sources[name] = matches[0]
    return sources


def inspect_bytes(path: Path, fallback_encoding: str | None) -> dict:
    """Hash every byte and validate UTF-8 incrementally, even in sampled mode."""
    digest = hashlib.sha256()
    decoder = codecs.getincrementaldecoder("utf-8-sig")("strict")
    valid_utf8 = True
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
            if valid_utf8:
                try:
                    decoder.decode(block)
                except UnicodeDecodeError:
                    valid_utf8 = False
        if valid_utf8:
            try:
                decoder.decode(b"", final=True)
            except UnicodeDecodeError:
                valid_utf8 = False
    if not valid_utf8 and fallback_encoding is None:
        raise ValueError(f"{path}: invalid UTF-8; configure a provisional \
                --fallback-encoding or confirm the source encoding")
    encoding = "utf-8-sig" if valid_utf8 else fallback_encoding
    if not valid_utf8:
        with path.open(encoding=encoding, errors="strict") as stream:
            while stream.read(1024 * 1024):
                pass
    return {
        "file": str(path),
        "bytes": size,
        "sha256": digest.hexdigest(),
        "encoding_used": encoding,
        "utf8_decodable": valid_utf8,
        "encoding_status": (
            "UTF-8 compatible; original encoding not proven"
            if valid_utf8
            else "Configured fallback; original encoding unconfirmed"
        ),
    }


def app_id(value: str) -> int | None:
    value = value.strip()
    if not re.fullmatch(r"[0-9]+", value):
        return None
    value = value.lstrip("0") or "0"
    if len(value) > 19:
        return None
    parsed = int(value)
    return parsed if 0 < parsed <= 2**63 - 1 else None


def rows(path: Path, encoding: str, header: list[str] | None = None):
    with path.open(encoding=encoding, errors="strict", newline="") as stream:
        reader = csv.reader(stream, strict=True)
        if header is not None and next(reader, None) != header:
            raise ValueError(f"{path}: expected header {header}")
        yield from reader


def history_chunks(
    path: Path,
    encoding: str,
    header: list[str],
    chunk_size: int,
    max_rows: int | None,
):
    """Validate CSV widths before building dataframes;
    never truncate extra cells."""
    with closing(rows(path, encoding, header)) as reader:
        selected = islice(reader, max_rows) if max_rows is not None else reader
        offset = 2
        while batch := list(islice(selected, chunk_size)):
            for record_number, row in enumerate(batch, start=offset):
                if len(row) != len(header):
                    raise ValueError(
                        f"{path}: unexpected row width in CSV record {record_number}: "
                        f"expected {len(header)} fields, found {len(row)}"
                    )
            yield pd.DataFrame(batch, columns=header, dtype=str)
            offset += len(batch)


def release_precision(value: str) -> str:
    if not value.strip():
        return "missing"
    for format_string, precision in (("%d-%b-%y", "day"), ("%b-%y", "month")):
        try:
            datetime.strptime(value.strip(), format_string)
            return precision
        except ValueError:
            pass
    return "unparsed"


def profile_information(path: Path, base: dict) -> tuple[dict, set[int]]:
    counts = Counter()
    identifiers = Counter()
    types, booleans, precisions = Counter(), Counter(), Counter()
    missing = Counter()
    examples = []
    header = HEADERS[INFORMATION]
    for row in rows(path, base["encoding_used"], header):
        counts["rows_scanned"] += 1
        if len(row) != len(header):
            counts["malformed_rows"] += 1
            continue
        record = dict(zip(header, row))
        for column, value in record.items():
            if not value.strip():
                missing[column] += 1
        identifier = app_id(record["appid"])
        if identifier is None:
            counts["invalid_app_id_rows"] += 1
        else:
            identifiers[identifier] += 1
        types[record["type"].strip() or "<blank>"] += 1
        booleans[record["freetoplay"].strip() or "<blank>"] += 1
        precision = release_precision(record["releasedate"])
        precisions[precision] += 1
        if precision in {"month", "unparsed"} and len(examples) < 5:
            examples.append(
                {
                    "app_id": identifier,
                    "raw": record["releasedate"],
                    "precision": precision,
                }
            )
    return base | {
        "kind": "application_snapshot",
        "scope": "complete",
        **counts,
        "grain": "One application in the supplied metadata snapshot",
        "expected_key": "appid (mapped to app_id)",
        "header_rule": ", ".join(header),
        "null_semantics": "Blank metadata is unknown; month-only release dates retain month precision",
        "distinct_applications": len(identifiers),
        "duplicate_key_rows": sum(count - 1 for count in identifiers.values()),
        "invalid_app_id_rows": counts["invalid_app_id_rows"],
        "malformed_rows": counts["malformed_rows"],
        "missing_by_column": dict(missing),
        "application_types": dict(types),
        "free_to_play_values": dict(booleans),
        "invalid_free_to_play_rows": sum(
            count
            for value, count in booleans.items()
            if value not in {"0", "1", "<blank>"}
        ),
        "release_date_precision": dict(precisions),
        "release_date_examples": examples,
    }, set(identifiers)


def profile_relationships(
    path: Path, base: dict, known_ids: set[int], attribute: str
) -> dict:
    counts, widths, identifiers = Counter(), Counter(), Counter()
    pairs = set()
    unknown_ids = set()
    for row in rows(path, base["encoding_used"]):
        counts["rows_scanned"] += 1
        widths[len(row)] += 1
        if not row or all(not value.strip() for value in row):
            counts["blank_rows"] += 1
            continue
        identifier = app_id(row[0])
        if identifier is None:
            counts["invalid_app_id_rows"] += 1
            continue
        identifiers[identifier] += 1
        if identifier not in known_ids:
            counts["orphan_rows"] += 1
            unknown_ids.add(identifier)
        values = [value.strip() for value in row[1:] if value.strip()]
        counts["blank_value_cells"] += len(row) - 1 - len(values)
        counts["id_only_rows"] += len(row) == 1
        counts["rows_without_values"] += not values
        for value in values:
            pair = (identifier, value)
            counts["relationship_values"] += 1
            counts["duplicate_relationship_rows"] += pair in pairs
            pairs.add(pair)
    return base | {
        "kind": "relationships",
        "scope": "complete",
        **counts,
        "grain": f"One application followed by zero or more {attribute} values",
        "expected_key": f"app_id per source row; (app_id, {attribute}) after expansion",
        "header_rule": "Headerless; first record is data; variable row width",
        "null_semantics": "No values may mean empty or unknown; unresolved",
        "distinct_applications": len(identifiers),
        "unique_relationships": len(pairs),
        "row_widths": dict(sorted(widths.items())),
        "duplicate_key_rows": sum(count - 1 for count in identifiers.values()),
        "missing_application_rows": len(known_ids - identifiers.keys()),
        "orphan_rows": counts["orphan_rows"],
        "orphan_app_ids": sorted(unknown_ids),
        "invalid_app_id_rows": counts["invalid_app_id_rows"],
    }


def numeric_profile(values: pd.Series) -> dict[str, int | float | None]:
    stripped = values.str.strip()
    missing = stripped.eq("")
    numbers = pd.to_numeric(stripped, errors="coerce")
    finite = numbers.notna() & ~numbers.isin([float("inf"), float("-inf")])
    valid = numbers[finite]
    return {
        "missing": int(missing.sum()),
        "invalid": int((~missing & ~finite).sum()),
        "zero": int(valid.eq(0).sum()),
        "negative": int(valid.lt(0).sum()),
        "fractional": int(valid.mod(1).ne(0).sum()),
        "minimum": float(valid.min()) if len(valid) else None,
        "maximum": float(valid.max()) if len(valid) else None,
    }


def merge_numeric(total: dict, part: dict) -> None:
    for key, value in part.items():
        if key in {"minimum", "maximum"}:
            previous = total.get(key)
            total[key] = (
                value
                if previous is None
                else (
                    previous
                    if value is None
                    else (
                        min(previous, value)
                        if key == "minimum"
                        else max(previous, value)
                    )
                )
            )
        else:
            total[key] = total.get(key, 0) + value


def profile_history(
    path: Path,
    name: str,
    base: dict,
    known_ids: set[int],
    chunk_size: int,
    max_rows: int | None,
    work_dir: Path | None,
) -> tuple[dict, list[dict]]:
    header = HEADERS[name]
    hourly = name == "Playercount_top1000.csv"
    prices = name == "Priceshistory.csv"
    time_column = "Date" if prices else "Time"
    time_format = "%Y-%m-%d %H:%M:%S" if hourly else "%Y-%m-%d"
    time_pattern = (
        r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}"
        if hourly
        else r"\d{4}-\d{2}-\d{2}"
    )
    counts = Counter(
        {
            key: 0
            for key in (
                "rows_scanned",
                "invalid_app_id_rows",
                "orphan_rows",
                "missing_time_rows",
                "invalid_time_rows",
                "valid_key_rows",
            )
        }
    )
    intervals = Counter()
    numeric = {column: {} for column in header[2:]}
    identifiers, unknown_ids = set(), set()
    last_time, coverage = {}, {}
    minimum = maximum = None
    duplicate_rows = unique_keys = 0
    with tempfile.TemporaryDirectory(
        prefix="xomnia-audit-", dir=work_dir
    ) as scratch:
        with closing(
            sqlite3.connect(str(Path(scratch) / "keys.sqlite"))
        ) as database:
            database.execute("PRAGMA journal_mode=OFF")
            database.execute("PRAGMA synchronous=OFF")
            database.execute("PRAGMA cache_size=-16384")
            database.execute(
                "CREATE TABLE keys (app_id INTEGER, time TEXT, PRIMARY KEY (app_id, time)) WITHOUT ROWID"
            )
            with closing(
                history_chunks(
                    path, base["encoding_used"], header, chunk_size, max_rows
                )
            ) as chunks:
                for chunk in chunks:
                    counts["rows_scanned"] += len(chunk)
                    raw_ids = chunk["app_id"].str.strip()
                    id_lookup = {
                        value: app_id(value) for value in raw_ids.unique()
                    }
                    ids = pd.Series(
                        pd.array(
                            [id_lookup[value] for value in raw_ids],
                            dtype="Int64",
                        ),
                        index=chunk.index,
                    )
                    valid_ids = ids.notna()
                    counts["invalid_app_id_rows"] += int((~valid_ids).sum())
                    identifiers.update(int(x) for x in ids.dropna().unique())
                    orphan = valid_ids & ~ids.isin(known_ids)
                    counts["orphan_rows"] += int(orphan.sum())
                    unknown_ids.update(int(x) for x in ids[orphan].unique())
                    raw_times = chunk[time_column].str.strip()
                    timestamps = pd.to_datetime(
                        raw_times.where(raw_times.str.fullmatch(time_pattern)),
                        format=time_format,
                        errors="coerce",
                    )
                    counts["missing_time_rows"] += int(raw_times.eq("").sum())
                    counts["invalid_time_rows"] += int(
                        (timestamps.isna() & raw_times.ne("")).sum()
                    )
                    if timestamps.notna().any():
                        lower, upper = timestamps.min(), timestamps.max()
                        minimum = (
                            lower if minimum is None else min(minimum, lower)
                        )
                        maximum = (
                            upper if maximum is None else max(maximum, upper)
                        )
                    for column in numeric:
                        merge_numeric(
                            numeric[column], numeric_profile(chunk[column])
                        )
                    if prices:
                        discount = pd.to_numeric(
                            chunk["Discount"], errors="coerce"
                        )
                        counts["discount_outside_0_100_rows"] += int(
                            (discount.lt(0) | discount.gt(100)).sum()
                        )
                    valid_keys = valid_ids & timestamps.notna()
                    counts["valid_key_rows"] += int(valid_keys.sum())
                    key_rows = pd.DataFrame(
                        {
                            "app_id": ids[valid_keys],
                            "time": raw_times[valid_keys],
                        }
                    )
                    before = database.total_changes
                    database.executemany(
                        "INSERT OR IGNORE INTO keys VALUES (?, ?)",
                        (
                            (int(identifier), time)
                            for identifier, time in key_rows.itertuples(
                                index=False, name=None
                            )
                        ),
                    )
                    inserted = database.total_changes - before
                    unique_keys += inserted
                    duplicate_rows += len(key_rows) - inserted
                    database.commit()

                    observations = pd.DataFrame(
                        {
                            "app_id": ids[valid_keys],
                            "time": timestamps[valid_keys],
                        }
                    )
                    grouped = observations.groupby("app_id", sort=False)
                    previous = grouped["time"].shift()
                    first = ~observations["app_id"].duplicated()
                    previous.loc[first] = pd.to_datetime(
                        observations.loc[first, "app_id"].map(last_time)
                    )
                    differences = (
                        (observations["time"] - previous)
                        .dt.total_seconds()
                        .dropna()
                    )
                    intervals.update(
                        {
                            int(key): int(value)
                            for key, value in differences.value_counts().items()
                        }
                    )
                    for identifier, group in grouped:
                        identifier = int(identifier)
                        last_time[identifier] = group["time"].iloc[-1]
                        item = coverage.setdefault(
                            identifier,
                            {
                                "rows": 0,
                                "start": group["time"].min(),
                                "end": group["time"].max(),
                            },
                        )
                        item["rows"] += len(group)
                        item["start"] = min(item["start"], group["time"].min())
                        item["end"] = max(item["end"], group["time"].max())
    result = base | {
        "kind": "price_history" if prices else "player_history",
        **counts,
        "scope": (
            f"first {max_rows:,} rows at most (not a random sample)"
            if max_rows
            else "complete"
        ),
        "grain": (
            "One application/date price observation in USD"
            if prices
            else "One application/time player-count observation"
        ),
        "expected_key": f"(app_id, {time_column}) within this source",
        "header_rule": ", ".join(header),
        "null_semantics": "Blank is unknown, distinct from an observed zero;\
            no filling or interpolation",
        "timezone": "unknown; timestamps remain naive",
        "expected_interval_seconds": 3600 if hourly else 86400,
        "coverage_start": str(minimum) if minimum is not None else None,
        "coverage_end": str(maximum) if maximum is not None else None,
        "distinct_applications": len(identifiers),
        "application_ids": sorted(identifiers),
        "orphan_app_ids": sorted(unknown_ids),
        "duplicate_key_rows": duplicate_rows,
        "unique_valid_keys": unique_keys,
        "numeric_columns": numeric,
        "intervals_seconds_in_source_order": dict(sorted(intervals.items())),
        "interval_method": "Consecutive valid keys per application in source\
            order, including chunk boundaries; count values may be missing",
        "out_of_order_intervals": sum(
            count for interval, count in intervals.items() if interval < 0
        ),
        "daily_aggregation": "Not performed; hourly observations are preserved\
            and daily measurement semantics remain unresolved",
    }
    coverage_rows = [
        {
            "source": path.name,
            "scope": result["scope"],
            "app_id": identifier,
            "rows_with_valid_keys": values["rows"],
            "start": str(values["start"]),
            "end": str(values["end"]),
        }
        for identifier, values in sorted(coverage.items())
    ]
    return result, coverage_rows


def cell(value) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def write_csv(path: Path, records: list[dict], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=columns, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(records)


def analyze(
    data_dir: Path,
    output_dir: Path,
    *,
    chunk_size: int = 100_000,
    max_rows: int | None = None,
    fallback_encoding: str | None = "cp1252",
    work_dir: Path | None = None,
) -> dict:
    if chunk_size <= 0 or (max_rows is not None and max_rows <= 0):
        raise ValueError("chunk_size and max_rows must be positive")
    sources = discover_sources(data_dir)
    results, application_coverage = [], []
    known_ids = set()
    for name, path in sources.items():
        print(f"Analyzing {path.name} ...", flush=True)
        base = inspect_bytes(path, fallback_encoding)
        if name == INFORMATION:
            result, known_ids = profile_information(path, base)
        elif name in RELATIONSHIPS:
            result = profile_relationships(
                path, base, known_ids, RELATIONSHIPS[name]
            )
        else:
            result, coverage = profile_history(
                path, name, base, known_ids, chunk_size, max_rows, work_dir
            )
            application_coverage.extend(coverage)
        result["source_name"] = name
        results.append(result)
        print(
            f"  {result.get('rows_scanned', 0):,} rows; \
                {result['duplicate_key_rows']:,}  \
                    duplicate key rows; {result['scope']}",
            flush=True,
        )
    history_ids = {
        source["source_name"]: set(source["application_ids"])
        for source in results
        if "application_ids" in source
    }
    top, bottom, prices = (history_ids[name] for name in list(HEADERS)[1:])
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "settings": {
            "chunk_size": chunk_size,
            "max_rows_per_history": max_rows,
            "fallback_encoding": fallback_encoding,
        },
        "total_bytes": sum(source["bytes"] for source in results),
        "sources": results,
        "cohort_checks": {
            "top_bottom_overlap_ids": sorted(top & bottom),
            "metadata_ids_without_scanned_player_history": sorted(
                known_ids - top - bottom
            ),
            "price_ids_without_scanned_player_history": sorted(
                prices - top - bottom
            ),
            "player_ids_without_scanned_prices_count": len(
                (top | bottom) - prices
            ),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )

    write_csv(
        output_dir / "inventory.csv",
        results,
        [
            "source_name",
            "file",
            "bytes",
            "sha256",
            "scope",
            "rows_scanned",
            "grain",
            "expected_key",
            "header_rule",
            "encoding_used",
            "encoding_status",
            "null_semantics",
            "coverage_start",
            "coverage_end",
            "distinct_applications",
        ],
    )
    write_csv(
        output_dir / "application_coverage.csv",
        application_coverage,
        ["source", "scope", "app_id", "rows_with_valid_keys", "start", "end"],
    )
    return report


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "reports" / "step2"
    )
    parser.add_argument("--chunk-size", type=positive_int, default=100_000)
    parser.add_argument(
        "--max-rows",
        type=positive_int,
        help="Read at most this many rows per history; all metadata is still\
            scanned completely",
    )
    parser.add_argument(
        "--fallback-encoding",
        default="cp1252",
        help="Provisional fallback for non-UTF-8 input (default: cp1252);\
            never silently replaces characters",
    )
    parser.add_argument(
        "--strict-utf8", action="store_true", help="Reject non-UTF-8 input"
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        help="Existing directory for the temporary SQLite index;\
            allow disk space comparable to source size",
    )
    args = parser.parse_args()
    try:
        analyze(
            args.data_dir,
            args.output_dir,
            chunk_size=args.chunk_size,
            max_rows=args.max_rows,
            fallback_encoding=(
                None if args.strict_utf8 else args.fallback_encoding
            ),
            work_dir=args.work_dir,
        )
    except (
        ValueError,
        OSError,
        UnicodeError,
        csv.Error,
        sqlite3.Error,
    ) as error:
        parser.exit(1, f"Analysis failed: {error}\n")
    print(f"Reports written to {args.output_dir}")


if __name__ == "__main__":
    main()
