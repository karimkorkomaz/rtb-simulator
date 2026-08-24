"""
Tests for the feature allowlist in `backend/src/ingest/schema.py`
(`schema.feature_columns()`), the structural guard against leaking
post-auction / non-feature columns (`bidprice`, `payprice`, `logtype`,
`keypage`, ...) into a CTR model.

Deliberately stdlib `unittest`, not pytest: pytest is not installed and not
in `backend/requirements.txt` (see task constraints), so this must run with
zero new dependencies via:

    python -m unittest discover -s backend/tests -t .

run from the repository root. `-t .` sets the top-level import directory to
the repo root so `backend.src.ingest.schema` resolves as a package (there is
no `backend/__init__.py`; Python 3's implicit namespace packages make this
work as long as the repo root, not `backend/`, is on `sys.path`/the
discovery top-level).

`unittest.TestCase` is used throughout (not a home-grown assert style) so
pytest can also collect this file unmodified if pytest is ever added later.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

# Ensure the repo root is importable as `backend.src...` regardless of the
# cwd the test runner was invoked from -- verified empirically (see module
# docstring) rather than assumed, because `backend/` has no `__init__.py`
# and relies on Python 3 namespace-package resolution, which only works if
# the *repo root* (not `backend/`) is what's on sys.path.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.src.ingest import paths, schema  # noqa: E402


# The four columns the audit specifically named, plus the two derived cases
# (`keypage_hash`, `bidid`) that must also never surface. Kept as a
# constant so every test below asserts against the same literal list rather
# than each re-typing it slightly differently.
MUST_NEVER_BE_FEATURES = {
    "bidprice", "payprice", "logtype", "keypage", "keypage_hash", "bidid",
}

# A hardcoded stand-in for the real `impressions` Parquet output schema
# (see transform.output_schema("imp")), used so requirement 3 ("run against
# the real schema, not just a hand-written list") has a companion case that
# ALWAYS runs, even on a clean clone with no ingested data. Deliberately
# kept in sync by construction with schema.py's own building blocks rather
# than typed out independently, so it can't silently drift from the real
# schema through a copy-paste-and-forget: it's IMP_CLK_CONV_COLUMNS (the
# documented 24-column layout) plus the documented hash companions and
# partition keys, which is exactly the shape of the real table.
HARDCODED_IMPRESSIONS_SCHEMA = (
    list(schema.IMP_CLK_CONV_COLUMNS)
    + [f"{base}_hash" for base in schema.HIGH_CARDINALITY_HASH_FIELDS]
    + list(schema.PARTITION_KEY_COLUMNS)
)


def _assert_structurally_admissible(testcase: unittest.TestCase, features: list) -> None:
    """Requirement 2: assert the STRUCTURAL property, not just the four
    literal names -- every returned feature must be a genuine member of
    BID_COLUMNS, or the `_hash` companion of one. This is what catches a
    *future* column silently becoming a feature (e.g. some new field added
    to the raw logs next season) even though nobody wrote a test naming it
    specifically.
    """
    bid_columns = set(schema.BID_COLUMNS)
    for feature in features:
        if feature.endswith("_hash"):
            base = feature[: -len("_hash")]
            testcase.assertIn(
                base, bid_columns,
                f"{feature!r} is a hash companion of {base!r}, which is not "
                "in BID_COLUMNS -- it should have been excluded.",
            )
        else:
            testcase.assertIn(
                feature, bid_columns,
                f"{feature!r} is not a member of BID_COLUMNS at all -- it "
                "should never have been admitted as a feature.",
            )


class TestFeatureColumnsHardcodedSchema(unittest.TestCase):
    """Requirement 3 (the always-runs half): a hardcoded column-list case
    that does not depend on the real dataset being present.
    """

    def setUp(self):
        self.features = schema.feature_columns(HARDCODED_IMPRESSIONS_SCHEMA)

    def test_named_leak_columns_absent(self):
        # Requirement 1: the exact assertion the audit asked for.
        for col in MUST_NEVER_BE_FEATURES:
            self.assertNotIn(
                col, self.features,
                f"{col!r} must never appear in a returned feature set.",
            )

    def test_every_feature_is_structurally_admissible(self):
        # Requirement 2.
        self.assertTrue(self.features, "expected a non-empty feature list")
        _assert_structurally_admissible(self, self.features)

    def test_partition_keys_excluded_not_raised(self):
        # Requirement 5, pinned against the hardcoded schema: season/date
        # are present in HARDCODED_IMPRESSIONS_SCHEMA and must be silently
        # excluded (not raise, not appear as features).
        self.assertNotIn("season", self.features)
        self.assertNotIn("date", self.features)
        # Calling feature_columns() on a list containing them must not raise.
        schema.feature_columns(["season", "date"])  # no exception

    def test_ipinyouid_hash_admitted_but_keypage_hash_not(self):
        # A concrete companion-column pair check: ipinyouid IS in
        # BID_COLUMNS (bid-time signal) so its hash companion is admitted;
        # keypage is NOT in BID_COLUMNS (post-auction field) so its hash
        # companion is excluded -- this is the "good test case" the task
        # called out explicitly, checked without any name-based special
        # case in the implementation itself.
        self.assertIn("ipinyouid_hash", self.features)
        self.assertNotIn("keypage_hash", self.features)


@unittest.skipUnless(
    (paths.PROCESSED_ROOT / "impressions").exists(),
    "backend/data/processed/impressions is git-ignored and not present on "
    "a clean clone -- run `python -m src.ingest.ingest --sample` from "
    "backend/ to generate it and enable this test.",
)
class TestFeatureColumnsRealParquetSchema(unittest.TestCase):
    """Requirement 3 (the real-schema half): read the actual column names
    out of the ingested `impressions` Parquet output (not a hand-written
    stand-in) and run them through feature_columns(). Guarded with
    `skipUnless` so the suite still passes on a clean clone where
    `data/processed/` (git-ignored) doesn't exist yet.
    """

    @classmethod
    def setUpClass(cls):
        import pyarrow.parquet as pq

        part_files = sorted((paths.PROCESSED_ROOT / "impressions").rglob("*.parquet"))
        if not part_files:
            raise unittest.SkipTest("impressions/ exists but contains no .parquet files")
        real_schema = pq.ParquetFile(part_files[0]).schema_arrow
        cls.real_columns = list(real_schema.names) + list(schema.PARTITION_KEY_COLUMNS)
        cls.features = schema.feature_columns(cls.real_columns)

    def test_named_leak_columns_absent(self):
        for col in MUST_NEVER_BE_FEATURES:
            self.assertNotIn(col, self.features)

    def test_every_feature_is_structurally_admissible(self):
        self.assertTrue(self.features, "expected a non-empty feature list")
        _assert_structurally_admissible(self, self.features)

    def test_does_not_raise_on_real_schema(self):
        # feature_columns() must handle every column the real ingestion
        # pipeline actually produces without raising -- if this ever
        # raises, it means the real schema drifted from what schema.py
        # knows about, which is exactly what this test exists to catch.
        schema.feature_columns(self.real_columns)  # no exception


class TestFeatureColumnsFailsClosed(unittest.TestCase):
    """Requirement 4: unrecognised / renamed / join-suffixed columns raise
    rather than silently passing through -- the specific bypass the audit
    identified (denylist exact-name matching defeated by any rename/join
    suffix).
    """

    def test_raises_on_unrecognised_column(self):
        with self.assertRaises(ValueError):
            schema.feature_columns(["totally_made_up_column"])

    def test_raises_on_join_suffixed_payprice(self):
        # e.g. what a pandas merge would produce if `payprice` collided
        # with another column during a join.
        with self.assertRaises(ValueError):
            schema.feature_columns(["payprice_1"])

    def test_raises_on_join_suffixed_timestamp(self):
        with self.assertRaises(ValueError):
            schema.feature_columns(["timestamp_1"])

    def test_raises_on_renamed_payprice(self):
        with self.assertRaises(ValueError):
            schema.feature_columns(["pay_price"])

    def test_error_message_names_offending_columns_and_next_step(self):
        with self.assertRaises(ValueError) as ctx:
            schema.feature_columns(["payprice_1", "totally_made_up"])
        message = str(ctx.exception)
        self.assertIn("payprice_1", message)
        self.assertIn("totally_made_up", message)
        # Must say what to do about it, not just that it failed.
        self.assertIn("BID_COLUMNS", message)
        self.assertIn("NON_FEATURE_BID_COLUMNS", message)

    def test_does_not_raise_on_legitimate_non_feature_columns(self):
        # Sanity check paired with the above: legitimate columns that are
        # simply not admissible (bidprice, payprice, logtype, keypage,
        # bidid, season, date) must NOT raise -- only genuinely unknown
        # names should. This pins the fail-closed check to "unrecognised",
        # not "recognised but excluded".
        try:
            schema.feature_columns(
                ["bidprice", "payprice", "logtype", "keypage", "bidid",
                 "season", "date"]
            )
        except ValueError as exc:  # pragma: no cover - failure path
            self.fail(f"feature_columns() raised on legitimate non-feature "
                      f"columns: {exc}")


class TestPartitionKeyHandling(unittest.TestCase):
    """Requirement 5: pin down exactly how `season`/`date` are handled --
    they are recognised explicitly and excluded (never raise, never appear
    as features), per PARTITION_KEY_COLUMNS.
    """

    def test_partition_keys_are_declared(self):
        self.assertEqual(schema.PARTITION_KEY_COLUMNS, frozenset({"season", "date"}))

    def test_partition_keys_excluded_from_output(self):
        out = schema.feature_columns(list(schema.BID_COLUMNS) + ["season", "date"])
        self.assertNotIn("season", out)
        self.assertNotIn("date", out)

    def test_partition_keys_alone_do_not_raise(self):
        self.assertEqual(schema.feature_columns(["season", "date"]), [])


class TestNonFeatureBidColumnsRationale(unittest.TestCase):
    """Pins the CRITICAL DISTINCTION from the task: `bidprice`/`bidid` are
    an EXPLICIT exclusion set (present in BID_COLUMNS, deliberately
    excluded); `payprice`/`logtype`/`keypage` are STRUCTURAL (absent from
    BID_COLUMNS itself, not re-listed anywhere as an exclusion).
    """

    def test_non_feature_bid_columns_is_exactly_bidid_and_bidprice(self):
        self.assertEqual(
            schema.NON_FEATURE_BID_COLUMNS, frozenset({"bidid", "bidprice"})
        )

    def test_payprice_logtype_keypage_are_not_in_bid_columns(self):
        # The structural guarantee: these three are simply absent from
        # BID_COLUMNS, not present-but-denylisted.
        for col in ("payprice", "logtype", "keypage"):
            self.assertNotIn(col, schema.BID_COLUMNS)

    def test_bidprice_is_present_in_bid_columns_but_still_excluded(self):
        # bidprice genuinely IS in BID_COLUMNS (known at bid-request time)
        # -- it only becomes non-admissible via the explicit exclusion set.
        self.assertIn("bidprice", schema.BID_COLUMNS)
        self.assertNotIn("bidprice", schema.feature_columns(list(schema.BID_COLUMNS)))


if __name__ == "__main__":
    unittest.main()
