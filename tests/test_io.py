"""Unit tests for :func:`bess_fleet.io.safe_to_parquet`.

The atomic-write contract:

* On success, the target file exists with the new content.
* The temp file does **not** persist after a successful write.
* If the underlying ``to_parquet`` raises, the temp file may be
  left on disk but the **target file is unaffected**.

These tests cover all three cases. The pipeline's idempotency
claim depends on this guarantee.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from bess_fleet.io import safe_to_parquet


@pytest.fixture()
def tiny_frame() -> pd.DataFrame:
    return pd.DataFrame({"system_id": ["ID16", "ID17"], "value": [1.0, 2.5]})


class TestSafeToParquetHappyPath:

    def test_writes_target_file(self, tiny_frame, tmp_path: Path) -> None:
        target = tmp_path / "out.parquet"
        safe_to_parquet(tiny_frame, target, index=False, compression="snappy")
        assert target.exists()

    def test_tmp_file_does_not_persist(self, tiny_frame, tmp_path: Path) -> None:
        """After a successful write, ``<path>.tmp`` should not exist —
        it's been renamed onto the target."""
        target = tmp_path / "out.parquet"
        safe_to_parquet(tiny_frame, target, index=False)
        assert not target.with_suffix(".parquet.tmp").exists()

    def test_content_round_trips(self, tiny_frame, tmp_path: Path) -> None:
        target = tmp_path / "out.parquet"
        safe_to_parquet(tiny_frame, target, index=False, compression="snappy")
        round_trip = pd.read_parquet(target)
        pd.testing.assert_frame_equal(round_trip, tiny_frame)

    def test_accepts_string_path(self, tiny_frame, tmp_path: Path) -> None:
        """Convenience — string paths work as well as Path objects."""
        target = tmp_path / "out.parquet"
        safe_to_parquet(tiny_frame, str(target), index=False)
        assert target.exists()


class TestSafeToParquetAtomicity:

    def test_existing_target_unchanged_on_write_failure(
        self, tiny_frame, tmp_path: Path,
    ) -> None:
        """If the underlying to_parquet raises mid-write, the
        previously-written target file must not be corrupted."""
        target = tmp_path / "out.parquet"

        # Establish a baseline with the original frame
        original_frame = pd.DataFrame({"system_id": ["ID14"], "value": [99.0]})
        safe_to_parquet(original_frame, target, index=False)
        baseline_bytes = target.read_bytes()

        # Now simulate a mid-write crash on the second write
        def boom(*_args, **_kwargs) -> None:
            raise OSError("simulated disk-full crash")

        with (
            patch.object(pd.DataFrame, "to_parquet", boom),
            pytest.raises(OSError, match="simulated disk-full"),
        ):
            safe_to_parquet(tiny_frame, target, index=False)

        # Target file must still contain the original baseline
        assert target.read_bytes() == baseline_bytes
        # And be readable as the original frame
        recovered = pd.read_parquet(target)
        pd.testing.assert_frame_equal(recovered, original_frame)

    def test_overwrites_existing_target_on_success(
        self, tiny_frame, tmp_path: Path,
    ) -> None:
        """The atomic-rename semantics must replace an existing target,
        not refuse to overwrite."""
        target = tmp_path / "out.parquet"
        original_frame = pd.DataFrame({"system_id": ["ID14"], "value": [99.0]})
        safe_to_parquet(original_frame, target, index=False)
        safe_to_parquet(tiny_frame, target, index=False)
        result = pd.read_parquet(target)
        pd.testing.assert_frame_equal(result, tiny_frame)
