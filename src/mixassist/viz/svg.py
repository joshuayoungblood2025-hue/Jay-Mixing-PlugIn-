"""Tiny inline-SVG builders (no external dependencies) used by the dashboard.

Each function returns an SVG fragment string. Coordinates are plain user units; the
dashboard wraps fragments in a sized ``<svg>`` element.
"""

from __future__ import annotations

from html import escape


def esc(s: str) -> str:
    return escape(str(s), quote=True)


def svg(width: float, height: float, body: str, extra: str = "") -> str:
    return (
        f'<svg viewBox="0 0 {width:.0f} {height:.0f}" width="100%" '
        f'preserveAspectRatio="xMidYMid meet" {extra} '
        f'xmlns="http://www.w3.org/2000/svg">{body}</svg>'
    )


def rect(x, y, w, h, fill, rx: float = 0.0, opacity: float = 1.0) -> str:
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(0.0, w):.1f}" height="{max(0.0, h):.1f}" '
        f'rx="{rx:.1f}" fill="{fill}" opacity="{opacity:.2f}"/>'
    )


def line(x1, y1, x2, y2, stroke, width: float = 1.0, dash: str = "") -> str:
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{stroke}" stroke-width="{width:.1f}"{d}/>'
    )


def text(
    x,
    y,
    s,
    fill: str = "#cbd5e1",
    size: float = 11.0,
    anchor: str = "start",
    weight: str = "normal",
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" fill="{fill}" font-size="{size:.0f}" '
        f'text-anchor="{anchor}" font-weight="{weight}" '
        f'font-family="ui-monospace,Menlo,Consolas,monospace">{esc(s)}</text>'
    )


def polyline(points: list[tuple[float, float]], stroke: str, width: float = 1.5) -> str:
    if not points:
        return ""
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return f'<polyline points="{pts}" fill="none" stroke="{stroke}" stroke-width="{width:.1f}"/>'


def circle(cx, cy, r, fill) -> str:
    return f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="{fill}"/>'
