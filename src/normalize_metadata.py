"""Normalize Steam's headerless metadata CSVs into UTF-8 relationship tables."""

import argparse
import csv
import io
import json
import tempfile
import warnings
from pathlib import Path

# Match only the six headerless files, not applicationInformation or histories.
METADATA_COLUMNS = {
    "applicationdevelopers.csv": "developer",
    "applicationpublishers.csv": "publisher",
    "applicationsupportedlanguages.csv": "language",
    "applicationpackages.csv": "package_id",
    "applicationgenres.csv": "genre",
    "applicationtags.csv": "tag",
}


def _decode_source(
    path: Path, fallback_encoding: str | None
) -> tuple[str, str, bool]:
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")  # Also accepts UTF-8 with a BOM.
        encoding = "utf-8-sig" if raw.startswith(b"\xef\xbb\xbf") else "utf-8"
        return text, encoding, True
    except UnicodeDecodeError as error:
        if fallback_encoding is None:
            raise ValueError(
                f"{path}: invalid UTF-8 at byte {error.start}. "
                "Supply fallback_encoding only if the source encoding is known."
            ) from error

        # This is a configured assumption, not automatic encoding detection.
        # Strict decoding prevents silently dropping or replacing characters.
        text = raw.decode(fallback_encoding)
        warnings.warn(
            f"{path.name} is not UTF-8; using configured fallback "
            f"{fallback_encoding!r}.",
            UnicodeWarning,
            stacklevel=3,
        )
        return text, fallback_encoding, False


def normalize_metadata_folder(
    input_folder: str | Path,
    output_folder: str | Path | None = None,
    *,
    fallback_encoding: str | None = "cp1252",
) -> list[dict[str, str | int | bool]]:

    input_path = Path(input_folder)
    if not input_path.is_dir():
        raise NotADirectoryError(f"Input folder does not exist: {input_path}")

    sources = [
        path
        for path in sorted(input_path.iterdir())
        if path.is_file() and path.name.lower() in METADATA_COLUMNS
    ]
    if not sources:
        raise ValueError(
            f"No supported headerless metadata CSVs in {input_path}"
        )

    output_path = (
        Path(output_folder)
        if output_folder is not None
        else input_path / "normalized"
    )
    output_path.mkdir(parents=True, exist_ok=True)
    reports = []

    for source in sources:
        text, encoding, input_is_utf8 = _decode_source(
            source, fallback_encoding
        )
        reader = csv.reader(io.StringIO(text, newline=""), strict=True)
        normalized_rows = []
        seen = set()
        input_rows = blank_rows = rows_without_values = duplicates_removed = 0

        try:
            for row in reader:
                if not row or all(not value.strip() for value in row):
                    blank_rows += 1
                    continue

                input_rows += 1
                try:
                    app_id = int(row[0].strip())
                    if app_id <= 0:
                        raise ValueError("app_id must be positive")
                except ValueError as error:
                    raise ValueError(
                        f"{source}, CSV line {reader.line_num}: expected a positive "
                        f"integer app_id, got {row[0]!r}. These files must be headerless."
                    ) from error

                values = [value.strip() for value in row[1:] if value.strip()]
                if not values:
                    rows_without_values += 1

                for value in values:
                    pair = (app_id, value)
                    if pair in seen:
                        duplicates_removed += 1
                        continue
                    seen.add(pair)
                    normalized_rows.append(pair)
        except csv.Error as error:
            raise ValueError(
                f"{source}, CSV line {reader.line_num}: {error}"
            ) from error

        destination = output_path / f"{source.stem}_normalized.csv"
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="",
                dir=output_path,
                prefix=f".{source.stem}_",
                suffix=".tmp",
                delete=False,
            ) as output:
                temporary_path = Path(output.name)
                writer = csv.writer(output)
                writer.writerow(
                    ["app_id", METADATA_COLUMNS[source.name.lower()]]
                )
                writer.writerows(normalized_rows)
            temporary_path.replace(destination)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

        reports.append(
            {
                "source": str(source),
                "input_is_utf8": input_is_utf8,
                "encoding_used": encoding,
                "input_rows": input_rows,
                "blank_rows": blank_rows,
                "rows_without_values": rows_without_values,
                "duplicates_removed": duplicates_removed,
                "output_rows": len(normalized_rows),
                "output": str(destination),
            }
        )

    return reports


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_folder", type=Path)
    parser.add_argument("--output", "-o", type=Path, default=None)
    parser.add_argument("--fallback-encoding", default="cp1252")
    parser.add_argument(
        "--strict-utf8",
        action="store_true",
        help="Reject non-UTF-8 sources instead of using the fallback encoding.",
    )
    args = parser.parse_args()
    report = normalize_metadata_folder(
        args.input_folder,
        args.output,
        fallback_encoding=None if args.strict_utf8 else args.fallback_encoding,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
