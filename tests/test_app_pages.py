"""Integration tests for the Streamlit pages.

These tests use Streamlit's official ``AppTest`` harness to spin up
each page in-process, run it to completion, and assert against the
rendered state. No browser, no Playwright — fast enough to live in
the regular pytest suite.

Requires the gold parquets (``data/curated/daily_kpis.parquet`` and
``data/curated/threshold_events.parquet``) to be present. If they're
not, the tests are skipped with a clear message — these are
integration tests, not unit tests, and they need real data shape.
Build the data with ``python bootstrap_data.py`` before running.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
APP_DIR = REPO_ROOT / "app"

# Pages import via ``from _components import ...`` which requires the
# ``app/`` folder on sys.path — Streamlit's normal launch adds it
# automatically; AppTest does not, so we add it explicitly here.
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

# `streamlit.testing` is the modern AppTest harness (Streamlit ≥ 1.28).
streamlit_testing = pytest.importorskip(
    "streamlit.testing.v1",
    reason="Streamlit < 1.28 has no AppTest harness",
)
AppTest = streamlit_testing.AppTest


DAILY_KPIS = REPO_ROOT / "data" / "curated" / "daily_kpis.parquet"
THRESHOLD_EVENTS = REPO_ROOT / "data" / "curated" / "threshold_events.parquet"
DEGRADATION = REPO_ROOT / "data" / "curated" / "degradation_modes.parquet"

DATA_PRESENT = DAILY_KPIS.exists() and THRESHOLD_EVENTS.exists()

needs_data = pytest.mark.skipif(
    not DATA_PRESENT,
    reason=(
        "Streamlit integration tests need gold parquets at "
        f"{DAILY_KPIS.parent} — run `python bootstrap_data.py` first."
    ),
)

needs_degradation = pytest.mark.skipif(
    not DEGRADATION.exists(),
    reason=(
        "Degradation page test needs "
        f"{DEGRADATION.name} — run "
        "`python -m bess_fleet.pipeline.degradation_modes` first."
    ),
)

CAPACITY_SOH = REPO_ROOT / "data" / "curated" / "capacity_soh.parquet"
needs_capacity = pytest.mark.skipif(
    not CAPACITY_SOH.exists(),
    reason=(
        f"SOHc page test needs {CAPACITY_SOH.name} — run "
        "`python -m bess_fleet.pipeline.capacity_estimation` first."
    ),
)


# ─── Fleet Overview page ─────────────────────────────────────────────


@needs_data
class TestFleetOverviewPage:
    """The Fleet Overview is the entry page — must render without
    exception and surface both systems in the status grid."""

    def test_page_runs_without_exception(self) -> None:
        at = AppTest.from_file(
            str(REPO_ROOT / "app" / "Fleet_Overview.py"),
            default_timeout=30,
        ).run()
        assert not at.exception, f"Page raised: {at.exception}"

    def test_renders_fleet_overview_heading(self) -> None:
        """The page title 'Fleet Overview' must be in the rendered HTML."""
        at = AppTest.from_file(
            str(REPO_ROOT / "app" / "Fleet_Overview.py"),
            default_timeout=30,
        ).run()
        rendered = " ".join(m.value for m in at.markdown)
        assert "Fleet Overview" in rendered

    def test_renders_both_systems_in_status_grid(self) -> None:
        """Both ID16 and ID17 must appear in the systems status table."""
        at = AppTest.from_file(
            str(REPO_ROOT / "app" / "Fleet_Overview.py"),
            default_timeout=30,
        ).run()
        rendered = " ".join(m.value for m in at.markdown)
        assert "ID16" in rendered
        assert "ID17" in rendered

    def test_id17_carries_watch_or_action_signal(self) -> None:
        """ID17 has a hard-coded notable-finding override that escalates
        its status to Watch. The page must surface that."""
        at = AppTest.from_file(
            str(REPO_ROOT / "app" / "Fleet_Overview.py"),
            default_timeout=30,
        ).run()
        rendered = " ".join(m.value for m in at.markdown).lower()
        # The page renders "Watch" status and/or "Action Recommended" pill
        assert ("watch" in rendered) or ("action recommended" in rendered)

    def test_status_grid_spans_all_chemistries(self) -> None:
        """With the scope widened to the full cross-chemistry fleet, the
        grid must surface NMC and LMO systems alongside the LFP racks."""
        at = AppTest.from_file(
            str(REPO_ROOT / "app" / "Fleet_Overview.py"),
            default_timeout=30,
        ).run()
        rendered = " ".join(m.value for m in at.markdown)
        assert "ID07" in rendered   # pure NMC (Mfr B)
        assert "ID01" in rendered   # LMO/NMC (Mfr A)


# ─── System deep-dive page ───────────────────────────────────────────


@needs_data
class TestSystemPage:
    """The System page renders one rack at a time. Default selection is
    ID16 (healthy baseline). The page must run without exception, render
    the deep-dive heading, and surface the identity strip."""

    PAGE = "app/pages/1_System.py"

    def test_page_runs_without_exception(self) -> None:
        at = AppTest.from_file(
            str(REPO_ROOT / self.PAGE),
            default_timeout=45,
        ).run()
        assert not at.exception, f"Page raised: {at.exception}"

    def test_renders_system_deep_dive_heading(self) -> None:
        at = AppTest.from_file(
            str(REPO_ROOT / self.PAGE),
            default_timeout=45,
        ).run()
        rendered = " ".join(m.value for m in at.markdown)
        assert "System deep dive" in rendered

    def test_system_selector_offers_both_systems(self) -> None:
        """The selectbox must contain both ID16 and ID17 as options."""
        at = AppTest.from_file(
            str(REPO_ROOT / self.PAGE),
            default_timeout=45,
        ).run()
        # The first selectbox is the System picker
        system_box = at.selectbox[0]
        options = [str(o) for o in system_box.options]
        assert any("ID16" in o for o in options)
        assert any("ID17" in o for o in options)

    def test_id17_renders_without_exception(self) -> None:
        """Run once (default ID16) to register the selectbox, then
        switch to ID17 and rerun. ID17 exercises the notable-finding
        code path that ID16 skips."""
        at = AppTest.from_file(
            str(REPO_ROOT / self.PAGE),
            default_timeout=45,
        ).run()
        assert not at.exception, f"Initial render raised: {at.exception}"

        at.selectbox[0].set_value("ID17").run()
        assert not at.exception, f"ID17 render raised: {at.exception}"

    def test_time_window_selector_has_four_options(self) -> None:
        """The Time Window picker offers 30 / 45 / 90 / 365 days."""
        at = AppTest.from_file(
            str(REPO_ROOT / self.PAGE),
            default_timeout=45,
        ).run()
        # Time window is the second selectbox on the page
        window_box = at.selectbox[1]
        labels = [str(o) for o in window_box.options]
        assert "Last 30 days" in labels
        assert "Last 1 year" in labels

    def test_renders_nmc_system_without_exception(self) -> None:
        """A NMC system (approximate SoC, different OCV + threshold limits)
        must render through the System page without raising — the cross-
        chemistry guarantee on this page."""
        at = AppTest.from_file(str(REPO_ROOT / self.PAGE), default_timeout=60).run()
        options = [str(o) for o in at.selectbox[0].options]
        assert any("ID07" in o for o in options)
        at.selectbox[0].set_value("ID07").run()
        assert not at.exception, f"NMC render raised: {at.exception}"


# ─── Degradation page ────────────────────────────────────────────────


@needs_degradation
class TestDegradationPage:
    """The cross-chemistry degradation page must render, surface the
    observability framing, and offer a per-system trend selector."""

    PAGE = "app/pages/2_Degradation.py"

    def test_page_runs_without_exception(self) -> None:
        at = AppTest.from_file(str(REPO_ROOT / self.PAGE), default_timeout=45).run()
        assert not at.exception, f"Page raised: {at.exception}"

    def test_renders_degradation_heading(self) -> None:
        at = AppTest.from_file(str(REPO_ROOT / self.PAGE), default_timeout=45).run()
        rendered = " ".join(m.value for m in at.markdown)
        assert "Degradation-mode estimation" in rendered

    def test_surfaces_observability_gate(self) -> None:
        """The observability framing is the headline — must be on the page."""
        at = AppTest.from_file(str(REPO_ROOT / self.PAGE), default_timeout=45).run()
        rendered = " ".join(m.value for m in at.markdown).lower()
        assert "observability" in rendered
        assert "lli" in rendered and "lam" in rendered

    def test_system_selector_present(self) -> None:
        at = AppTest.from_file(str(REPO_ROOT / self.PAGE), default_timeout=45).run()
        assert len(at.selectbox) >= 1
        options = [str(o) for o in at.selectbox[0].options]
        # At least one observable NMC-family system should be selectable.
        assert any("ID02" in o or "ID11" in o for o in options)


# ─── SOHc (capacity) page ────────────────────────────────────────────


@needs_capacity
class TestSohcPage:
    """The Figgener capacity page must render, surface the SOHc/ageing
    framing + reliability gate, and offer a per-system series selector."""

    PAGE = "app/pages/3_SOHc.py"

    def test_page_runs_without_exception(self) -> None:
        at = AppTest.from_file(str(REPO_ROOT / self.PAGE), default_timeout=45).run()
        assert not at.exception, f"Page raised: {at.exception}"

    def test_surfaces_sohc_and_reliability(self) -> None:
        at = AppTest.from_file(str(REPO_ROOT / self.PAGE), default_timeout=45).run()
        rendered = " ".join(m.value for m in at.markdown).lower()
        assert "sohc" in rendered
        assert "reliab" in rendered          # reliability gate framing
        assert "ageing" in rendered

    def test_reliable_system_selectable(self) -> None:
        at = AppTest.from_file(str(REPO_ROOT / self.PAGE), default_timeout=45).run()
        assert len(at.selectbox) >= 1
        options = [str(o) for o in at.selectbox[0].options]
        # A reliable LFP system (ID16) should be offered.
        assert any("ID16" in o for o in options)
