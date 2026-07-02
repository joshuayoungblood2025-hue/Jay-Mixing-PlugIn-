"""Render a self-contained HTML mixing dashboard (no external assets).

Everything — styles and charts (inline SVG) — is embedded in a single HTML string so it
can be opened offline or committed to a repo. Sections: loudness meters, frequency-balance
graph vs. genre target, short-term loudness timeline, stereo image, suggestions, and a
per-track breakdown.
"""

from __future__ import annotations

import math
from html import escape

from mixassist.analysis.metrics import MasterMetrics
from mixassist.analysis.spectrum import BAND_NAMES
from mixassist.mixing.engine import MixResult, MixSettings
from mixassist.mixing.suggestions import Suggestion
from mixassist.viz import svg

_BG = "#0f172a"
_PANEL = "#1e293b"
_TEXT = "#e2e8f0"
_MUTED = "#94a3b8"
_ACCENT = "#38bdf8"
_TARGET = "#f472b6"
_GOOD = "#22c55e"
_WARN = "#f59e0b"
_BAD = "#ef4444"
_SEV_COLOR = {"ok": _GOOD, "info": _ACCENT, "warn": _WARN}


def _fmt(x: float, unit: str = "") -> str:
    return f"{x:.1f}{unit}" if math.isfinite(x) else "--"


# --------------------------------------------------------------------------- charts


def _loudness_meters(m: MasterMetrics, target_lufs: float) -> str:
    """Horizontal meters on a -30..0 scale for loudness/peak values."""
    w, row_h, pad_l = 620.0, 30.0, 150.0
    rows = [
        ("Integrated", m.integrated_lufs, "LUFS", _ACCENT),
        ("Short-term max", m.short_term_max_lufs, "LUFS", _ACCENT),
        ("Momentary max", m.momentary_max_lufs, "LUFS", _ACCENT),
        ("Sample peak", m.peak_dbfs, "dBFS", _GOOD),
    ]
    lo, hi = -30.0, 0.0
    track_w = w - pad_l - 60.0

    def x_of(db: float) -> float:
        db = max(lo, min(hi, db))
        return pad_l + (db - lo) / (hi - lo) * track_w

    body = []
    height = row_h * (len(rows) + 1) + 20
    for i, (label, val, unit, color) in enumerate(rows):
        y = 20 + i * row_h
        body.append(svg.text(pad_l - 10, y + 12, label, fill=_MUTED, size=12, anchor="end"))
        body.append(svg.rect(pad_l, y, track_w, 16, "#0b1220", rx=3))
        if math.isfinite(val):
            body.append(svg.rect(pad_l, y, x_of(val) - pad_l, 16, color, rx=3, opacity=0.85))
        body.append(
            svg.text(pad_l + track_w + 8, y + 12, f"{_fmt(val)} {unit}", fill=_TEXT, size=12)
        )
    # target marker for integrated loudness
    ty = 20
    tx = x_of(target_lufs)
    body.append(svg.line(tx, ty - 4, tx, ty + row_h * len(rows) - 8, _TARGET, 1.5, dash="4,3"))
    body.append(
        svg.text(
            tx, height - 6, f"target {target_lufs:.1f}", fill=_TARGET, size=10, anchor="middle"
        )
    )
    # scale ticks
    for db in (-30, -24, -18, -12, -6, 0):
        xx = x_of(db)
        body.append(svg.text(xx, height - 18, str(db), fill="#475569", size=9, anchor="middle"))
    return svg.svg(w, height, "".join(body))


def _freq_balance(m: MasterMetrics, target_curve: list[float]) -> str:
    """Vertical bars of the mix tonal 'shape' (dB vs average) with target markers."""
    shape = m.tonal.normalized_db()
    n = len(BAND_NAMES)
    w, h = 620.0, 240.0
    pad_l, pad_b, pad_t = 34.0, 40.0, 16.0
    plot_h = h - pad_b - pad_t
    band_w = (w - pad_l - 12) / n
    rng = 18.0  # +/- dB shown

    def y_of(db: float) -> float:
        db = max(-rng, min(rng, db))
        return pad_t + (rng - db) / (2 * rng) * plot_h

    body = []
    zero_y = y_of(0.0)
    # gridlines
    for db in (-18, -12, -6, 0, 6, 12, 18):
        yy = y_of(db)
        body.append(svg.line(pad_l, yy, w - 6, yy, "#243044", 1))
        body.append(svg.text(pad_l - 6, yy + 3, f"{db:+d}", fill="#475569", size=9, anchor="end"))
    for i, name in enumerate(BAND_NAMES):
        val = shape[i] if i < len(shape) else 0.0
        tgt = target_curve[i] if i < len(target_curve) else 0.0
        cx = pad_l + i * band_w + band_w / 2
        by = y_of(val)
        color = _ACCENT
        if abs(val - tgt) > 4.0:
            color = _WARN
        top = min(by, zero_y)
        bh = abs(by - zero_y)
        body.append(svg.rect(cx - band_w * 0.32, top, band_w * 0.64, bh, color, rx=2, opacity=0.85))
        # target marker
        tyy = y_of(tgt)
        body.append(svg.line(cx - band_w * 0.4, tyy, cx + band_w * 0.4, tyy, _TARGET, 2.0))
        body.append(svg.text(cx, h - pad_b + 14, name[:4], fill=_MUTED, size=9, anchor="middle"))
    body.append(
        svg.text(w - 6, pad_t - 4, "bar = mix   — = target", fill=_TARGET, size=10, anchor="end")
    )
    return svg.svg(w, h, "".join(body))


def _timeline(m: MasterMetrics, target_lufs: float) -> str:
    w, h = 620.0, 150.0
    pad_l, pad_b, pad_t = 40.0, 24.0, 14.0
    plot_w = w - pad_l - 12
    plot_h = h - pad_b - pad_t
    series = [(t, v) for t, v in m.short_term_series if math.isfinite(v)]
    lo, hi = -30.0, 0.0
    body = []
    for db in (-30, -20, -10, 0):
        yy = pad_t + (hi - db) / (hi - lo) * plot_h
        body.append(svg.line(pad_l, yy, w - 6, yy, "#243044", 1))
        body.append(svg.text(pad_l - 6, yy + 3, str(db), fill="#475569", size=9, anchor="end"))
    ty = pad_t + (hi - target_lufs) / (hi - lo) * plot_h
    body.append(svg.line(pad_l, ty, w - 6, ty, _TARGET, 1.2, dash="4,3"))
    if series:
        tmax = max(t for t, _ in series) or 1.0
        pts = []
        for t, v in series:
            x = pad_l + (t / tmax) * plot_w if tmax else pad_l
            y = pad_t + (hi - max(lo, min(hi, v))) / (hi - lo) * plot_h
            pts.append((x, y))
        body.append(svg.polyline(pts, _ACCENT, 2.0))
        for x, y in pts:
            body.append(svg.circle(x, y, 2.2, _ACCENT))
    else:
        body.append(
            svg.text(
                w / 2,
                h / 2,
                "clip too short for short-term timeline",
                fill=_MUTED,
                size=11,
                anchor="middle",
            )
        )
    body.append(svg.text(pad_l, h - 6, "time -->", fill="#475569", size=9))
    return svg.svg(w, h, "".join(body))


def _stereo_panel(m: MasterMetrics) -> str:
    w, h = 620.0, 150.0
    body = []
    # correlation gauge (-1 .. +1)
    gx, gy, gw = 150.0, 34.0, 420.0
    body.append(svg.text(gx - 10, gy + 12, "Correlation", fill=_MUTED, size=12, anchor="end"))
    body.append(svg.rect(gx, gy, gw, 16, "#0b1220", rx=3))

    # zones: red left (<0), amber (0..0.3), green (>0.3)
    def cx_of(c: float) -> float:
        return gx + (max(-1.0, min(1.0, c)) + 1.0) / 2.0 * gw

    body.append(svg.rect(cx_of(-1), gy, cx_of(0) - cx_of(-1), 16, _BAD, rx=0, opacity=0.25))
    body.append(svg.rect(cx_of(0), gy, cx_of(0.3) - cx_of(0), 16, _WARN, rx=0, opacity=0.25))
    body.append(svg.rect(cx_of(0.3), gy, cx_of(1) - cx_of(0.3), 16, _GOOD, rx=0, opacity=0.25))
    mx = cx_of(m.stereo_correlation)
    body.append(svg.line(mx, gy - 4, mx, gy + 20, _TEXT, 2.0))
    body.append(
        svg.text(mx, gy - 8, f"{m.stereo_correlation:+.2f}", fill=_TEXT, size=11, anchor="middle")
    )
    for c in (-1.0, 0.0, 1.0):
        body.append(
            svg.text(cx_of(c), gy + 34, f"{c:+.0f}", fill="#475569", size=9, anchor="middle")
        )

    # width bars per band
    bx, by, bw = 150.0, 86.0, 420.0
    bands = [("low", m.width_low), ("mid", m.width_mid), ("high", m.width_high)]
    maxw = 1.5
    body.append(svg.text(bx - 10, by + 12, "Width", fill=_MUTED, size=12, anchor="end"))
    for i, (label, val) in enumerate(bands):
        yy = by + i * 18
        color = _ACCENT if not (label == "low" and val > 0.6) else _WARN
        body.append(svg.rect(bx, yy, bw, 12, "#0b1220", rx=2))
        body.append(svg.rect(bx, yy, min(1.0, val / maxw) * bw, 12, color, rx=2, opacity=0.85))
        body.append(svg.text(bx + bw + 8, yy + 10, f"{label} {val:.2f}", fill=_TEXT, size=10))
    return svg.svg(w, h, "".join(body))


# --------------------------------------------------------------------------- HTML


def _panel(title: str, inner: str, subtitle: str = "") -> str:
    sub = f'<div class="sub">{escape(subtitle)}</div>' if subtitle else ""
    return f'<section class="panel"><h2>{escape(title)}</h2>{sub}{inner}</section>'


def _suggestions_html(suggestions: list[Suggestion]) -> str:
    if not suggestions:
        return '<p class="muted">No suggestions — the mix meets the targets.</p>'
    items = []
    for s in suggestions:
        color = _SEV_COLOR.get(s.severity, _MUTED)
        badge = (
            f'<span class="badge" style="background:{color}">{escape(s.severity.upper())}</span>'
        )
        items.append(
            f"<li>{badge}"
            f'<span class="area">{escape(s.area)}</span>'
            f'<span class="msg">{escape(s.message)}</span></li>'
        )
    return f'<ul class="suggestions">{"".join(items)}</ul>'


def _tracks_html(result: MixResult) -> str:
    rows = []
    for p in result.track_plans:
        c = p.classification
        eq = " · ".join(b.describe() for b in p.eq_bands) if p.eq_bands else "—"
        comp = f"{p.comp.ratio:.1f}:1 (GR {p.comp_gr_db:.1f})" if p.comp else "—"
        flags = []
        if p.muted:
            flags.append('<span class="flag mute">MUTE</span>')
        if p.locked:
            flags.append('<span class="flag lock">LOCK</span>')
        flag_html = " ".join(flags)
        rows.append(
            "<tr>"
            f'<td class="name">{escape(p.name)} {flag_html}</td>'
            f'<td>{escape(c.role)}<span class="muted">/{escape(c.subtype)}</span></td>'
            f'<td class="num">{c.confidence:.2f}</td>'
            f'<td class="num">{p.gain_db:+.1f}</td>'
            f'<td class="num">{p.pan:+.2f}</td>'
            f'<td class="num">{p.width:.2f}</td>'
            f'<td class="num">{comp}</td>'
            f'<td class="eq">{escape(eq)}</td>'
            "</tr>"
        )
    return (
        '<table class="tracks"><thead><tr>'
        "<th>Track</th><th>Role</th><th>Conf</th><th>Gain</th><th>Pan</th>"
        "<th>Width</th><th>Comp</th><th>EQ moves</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )


def _header(result: MixResult, m: MasterMetrics, s: MixSettings) -> str:
    tone = "neutral"
    if s.tone > 0.02:
        tone = f"clarity +{s.tone:.2f}"
    elif s.tone < -0.02:
        tone = f"warmth {s.tone:.2f}"
    chips = [
        ("Genre", result.target.name),
        ("Integrated", f"{_fmt(m.integrated_lufs)} LUFS"),
        ("Peak", f"{_fmt(m.peak_dbfs)} dBFS"),
        ("PLR", f"{_fmt(m.plr_db)} dB"),
        ("Intensity", f"{s.intensity:.2f}"),
        ("Vocal", f"{s.vocal_prominence:.2f}"),
        ("Tone", tone),
    ]
    chip_html = "".join(
        f'<div class="chip"><span class="k">{escape(k)}</span>'
        f'<span class="v">{escape(str(v))}</span></div>'
        for k, v in chips
    )
    return (
        '<header><h1>AI Mixing Assistant <span class="tag">dashboard</span></h1>'
        f'<div class="chips">{chip_html}</div>'
        f'<p class="notes">{escape(result.target.notes)}</p></header>'
    )


_CSS = (
    """
* { box-sizing: border-box; }
body { margin:0; background:__BG__; color:__TEXT__;
  font-family: ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif; }
.wrap { max-width: 720px; margin: 0 auto; padding: 24px 16px 64px; }
header h1 { font-size: 20px; margin:0 0 12px; font-weight:700; }
header .tag { font-size:11px; color:__BG__; background:__ACCENT__; padding:2px 8px;
  border-radius:10px; vertical-align:middle; margin-left:6px; }
.chips { display:flex; flex-wrap:wrap; gap:8px; }
.chip { background:__PANEL__; border-radius:8px; padding:6px 10px; font-size:12px; }
.chip .k { color:__MUTED__; margin-right:6px; }
.chip .v { font-weight:600; }
.notes { color:__MUTED__; font-size:13px; margin:10px 0 0; }
.panel { background:__PANEL__; border-radius:12px; padding:16px; margin-top:18px; }
.panel h2 { font-size:14px; margin:0 0 10px; color:__TEXT__; letter-spacing:.02em; }
.panel .sub { color:__MUTED__; font-size:12px; margin:-4px 0 10px; }
.muted { color:__MUTED__; }
ul.suggestions { list-style:none; margin:0; padding:0; }
ul.suggestions li { display:flex; align-items:flex-start; gap:8px; padding:8px 0;
  border-bottom:1px solid #243044; font-size:13px; }
ul.suggestions li:last-child { border-bottom:none; }
.badge { color:__BG__; font-size:10px; font-weight:700; padding:2px 6px; border-radius:6px;
  min-width:42px; text-align:center; }
.area { color:__MUTED__; min-width:96px; text-transform:capitalize; }
.msg { flex:1; }
table.tracks { width:100%; border-collapse:collapse; font-size:12px; }
table.tracks th { text-align:left; color:__MUTED__; font-weight:600; padding:6px 8px;
  border-bottom:1px solid #334155; }
table.tracks td { padding:6px 8px; border-bottom:1px solid #1f2a3a; vertical-align:top; }
td.num { text-align:right; font-variant-numeric:tabular-nums; }
td.name { font-weight:600; }
td.eq { color:__MUTED__; max-width:220px; }
td.muted, .muted { color:__MUTED__; }
.flag { font-size:9px; padding:1px 5px; border-radius:5px; margin-left:4px; }
.flag.mute { background:__BAD__; color:__BG__; }
.flag.lock { background:__MUTED__; color:__BG__; }
""".replace("__BG__", _BG)
    .replace("__PANEL__", _PANEL)
    .replace("__TEXT__", _TEXT)
    .replace("__MUTED__", _MUTED)
    .replace("__ACCENT__", _ACCENT)
    .replace("__BAD__", _BAD)
)


def render_dashboard(
    result: MixResult,
    metrics: MasterMetrics,
    suggestions: list[Suggestion],
    settings: MixSettings,
) -> str:
    target_lufs = result.bus_plan.target_lufs
    target_curve = result.target.curve_list()
    body = "".join(
        [
            '<div class="wrap">',
            _header(result, metrics, settings),
            _panel("Loudness & peak", _loudness_meters(metrics, target_lufs)),
            _panel(
                "Frequency balance",
                _freq_balance(metrics, target_curve),
                "Mix tonal shape (dB vs. average) against the genre target curve.",
            ),
            _panel("Short-term loudness", _timeline(metrics, target_lufs)),
            _panel("Stereo image", _stereo_panel(metrics)),
            _panel("Suggestions", _suggestions_html(suggestions)),
            _panel("Track breakdown", _tracks_html(result)),
            "</div>",
        ]
    )
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        "<title>AI Mixing Assistant — Dashboard</title>"
        f"<style>{_CSS}</style></head><body>{body}</body></html>"
    )
