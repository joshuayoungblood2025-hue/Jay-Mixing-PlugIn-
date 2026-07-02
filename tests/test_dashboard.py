"""HTML dashboard generation."""

from __future__ import annotations

from mixassist.analysis.metrics import compute_master_metrics
from mixassist.mixing.engine import MixSettings, mix
from mixassist.mixing.suggestions import generate_suggestions
from mixassist.viz.dashboard import render_dashboard


def _build(synth_stems, settings):
    res = mix(synth_stems, settings)
    metrics = compute_master_metrics(res.master)
    suggestions = generate_suggestions(res, metrics)
    return render_dashboard(res, metrics, suggestions, settings)


def test_dashboard_is_valid_html(synth_stems):
    html = _build(synth_stems, MixSettings(genre="pop", intensity=0.6))
    assert html.lower().startswith("<!doctype html>")
    assert "</html>" in html


def test_dashboard_has_all_charts(synth_stems):
    html = _build(synth_stems, MixSettings(genre="pop"))
    # loudness meters, freq balance, timeline, stereo -> at least 4 SVGs
    assert html.count("<svg") >= 4


def test_dashboard_has_sections(synth_stems):
    html = _build(synth_stems, MixSettings(genre="pop"))
    for label in (
        "Loudness",
        "Frequency balance",
        "Stereo image",
        "Suggestions",
        "Track breakdown",
    ):
        assert label in html


def test_dashboard_has_no_unfilled_format_placeholders(synth_stems):
    html = _build(synth_stems, MixSettings(genre="pop"))
    assert "%(" not in html  # CSS %-format keys must all be resolved


def test_dashboard_escapes_track_names(synth_stems):
    stems = dict(synth_stems)
    stems["<script>evil</script>"] = stems.pop("Pad")
    html = _build(stems, MixSettings(genre="pop"))
    assert "<script>evil" not in html
    assert "&lt;script&gt;" in html
