"""Regression checks for evidence-changing edge cases in the source audit."""

import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path

from src.analyze_sources import (
    HEADERS,
    INFORMATION,
    RELATIONSHIPS,
    analyze,
    inspect_bytes,
    profile_history,
    profile_relationships,
)


class SourceAuditTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def write_csv(self, name, records, encoding="utf-8"):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding=encoding, newline="") as stream:
            csv.writer(stream).writerows(records)
        return path

    def history(self, path, chunk_size=2, max_rows=None, known_ids=None):
        return profile_history(path, path.name, inspect_bytes(path, None),
                               {1, 2} if known_ids is None else known_ids,
                               chunk_size, max_rows, self.root)

    def test_duplicates_nulls_and_intervals_across_chunks(self):
        name = "Playercount_top1000.csv"
        path = self.write_csv(name, [HEADERS[name],
            ["1", "2017-12-14 00:00:00", "0"],
            ["1", "2017-12-14 01:00:00", ""],
            ["2", "2017-12-14 00:00:00", "5.5"],
            ["01", "2017-12-14 00:00:00", "3"],
            ["999", "2017-12-14 00:00:00", "-2"],
            ["bad", "2017-12-14 02:00:00", "abc"],
            ["1", "not-a-date", "7"],
            ["1", "", "inf"],
        ])
        small, small_coverage = self.history(path, chunk_size=2)
        large, large_coverage = self.history(path, chunk_size=100)
        self.assertEqual(small, large)
        self.assertEqual(small_coverage, large_coverage)
        self.assertEqual(small["rows_scanned"], 8)
        self.assertEqual(small["duplicate_key_rows"], 1)
        self.assertEqual(small["unique_valid_keys"], 4)
        self.assertEqual(small["invalid_app_id_rows"], 1)
        self.assertEqual(small["invalid_time_rows"], 1)
        self.assertEqual(small["missing_time_rows"], 1)
        self.assertEqual(small["orphan_rows"], 1)
        self.assertEqual(small["intervals_seconds_in_source_order"], {-3600: 1, 3600: 1})
        numeric = small["numeric_columns"]["Playercount"]
        self.assertEqual({key: numeric[key] for key in ("zero", "missing", "invalid", "negative", "fractional")},
                         {"zero": 1, "missing": 1, "invalid": 2, "negative": 1, "fractional": 1})

    def test_ids_are_not_rounded_through_floating_point(self):
        name = "Playercount_bottom1000.csv"
        identifier = 9007199254740993
        path = self.write_csv(name, [HEADERS[name],
            [str(identifier), "2017-12-14", "1"], ["bad", "2017-12-14", "1"],
        ])
        result, _ = self.history(path, known_ids={identifier})
        self.assertEqual(result["application_ids"], [identifier])
        self.assertEqual(result["orphan_rows"], 0)

    def test_headerless_first_row_and_absent_lists(self):
        path = self.write_csv("applicationTags.csv", [
            ["1", "Action", " Action "], ["2"], ["2", ""], ["999", "Adventure"], [],
        ])
        result = profile_relationships(path, inspect_bytes(path, None), {1, 2}, "tag")
        self.assertEqual(result["unique_relationships"], 2)
        self.assertEqual(result["duplicate_relationship_rows"], 1)
        self.assertEqual(result["duplicate_key_rows"], 1)
        self.assertEqual(result["rows_without_values"], 2)
        self.assertEqual(result["id_only_rows"], 1)
        self.assertEqual(result["blank_value_cells"], 1)
        self.assertEqual(result["orphan_rows"], 1)
        self.assertEqual(result["blank_rows"], 1)

    def test_strict_encoding_and_provisional_fallback(self):
        path = self.write_csv("legacy.csv", [["1", "Café"]], encoding="cp1252")
        with self.assertRaisesRegex(ValueError, "invalid UTF-8"):
            inspect_bytes(path, None)
        result = inspect_bytes(path, "cp1252")
        self.assertFalse(result["utf8_decodable"])
        self.assertIn("unconfirmed", result["encoding_status"])
        path.write_bytes(b"1,\x81\n")
        with self.assertRaises(UnicodeDecodeError):
            inspect_bytes(path, "cp1252")

    def test_utf8_character_split_across_byte_chunks(self):
        path = self.root / "split.csv"
        path.write_bytes(b"a" * (1024 * 1024 - 1) + "é".encode("utf-8"))
        self.assertTrue(inspect_bytes(path, None)["utf8_decodable"])

    def test_unexpected_header_and_excess_columns_fail(self):
        name = "Playercount_bottom1000.csv"
        path = self.write_csv(name, [["appid", "Time", "Playercount"], ["1", "2017-12-14", "1"]])
        with self.assertRaisesRegex(ValueError, "expected header"):
            self.history(path)
        self.write_csv(name, [HEADERS[name], ["1", "2017-12-14", "1", "extra"]])
        with self.assertRaisesRegex(ValueError, "unexpected row width"):
            self.history(path)
        # Some chunked CSV parsers silently truncate excess fields after a boundary.
        self.write_csv(name, [HEADERS[name], ["1", "2017-12-14", "1"],
                             ["2", "2017-12-15", "2", "extra"]])
        with self.assertRaisesRegex(ValueError, "unexpected row width"):
            self.history(path, chunk_size=1)
        self.write_csv(name, [HEADERS[name], ["1", "2017-12-14"]])
        with self.assertRaisesRegex(ValueError, "unexpected row width"):
            self.history(path)

    def test_end_to_end_sample_reports_and_original_bytes(self):
        self.write_csv(INFORMATION, [HEADERS[INFORMATION],
            ["1", "game", "First", "Dec-17", "0"], ["2", "game", "Second", "14-Dec-17", "1"],
        ])
        for name, header in list(HEADERS.items())[1:]:
            date = "2017-12-14 00:00:00" if "top1000" in name else "2017-12-14"
            values = ["10.00", "5.00", "50"] if "Prices" in name else ["0"]
            self.write_csv(name, [header, ["1", date, *values], ["2", date, *values]])
        for name in RELATIONSHIPS:
            self.write_csv(f"raw/{name}", [["1", "attribute"], ["2"]])
        # A normalized copy with a matching filename must not become an input.
        self.write_csv("raw/normalized/applicationTags.csv", [["app_id", "tag"], ["999", "unexpected"]])
        originals = {path: path.read_bytes() for path in self.root.rglob("*.csv")}
        with contextlib.redirect_stdout(io.StringIO()):
            report = analyze(self.root, self.root / "report", chunk_size=1, max_rows=1)
        for path, content in originals.items():
            self.assertEqual(path.read_bytes(), content)
        self.assertEqual(len(report["sources"]), 10)
        information = report["sources"][0]
        self.assertEqual(information["scope"], "complete")
        self.assertEqual(information["release_date_precision"], {"month": 1, "day": 1})
        self.assertEqual(information["release_date_examples"][0]["raw"], "Dec-17")
        for source in report["sources"][1:4]:
            self.assertIn("not a random sample", source["scope"])
            self.assertEqual(source["rows_scanned"], 1)
        saved = json.loads((self.root / "report/report.json").read_text())
        self.assertEqual(saved["settings"]["max_rows_per_history"], 1)
        for filename in ("report.md", "inventory.csv", "application_coverage.csv"):
            self.assertTrue((self.root / "report" / filename).is_file())


if __name__ == "__main__":
    unittest.main()
