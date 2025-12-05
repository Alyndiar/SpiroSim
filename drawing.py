from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple

from PySide6.QtCore import QPointF
from PySide6.QtGui import QColor, QPainter, QPen

import modular_tracks_2 as modular_tracks

Point = Tuple[float, float]


def compute_view_scale(points: Sequence[Point], width: int, height: int, margin_ratio: float = 0.45) -> float:
    """Compute a uniform scale so that ``points`` fit inside the viewport.

    The returned scale keeps aspect ratio, applies the requested margin, and
    returns ``1.0`` when there is nothing to draw.
    """

    if not points:
        return 1.0

    max_x = max(abs(x) for x, _ in points) or 1.0
    max_y = max(abs(y) for _, y in points) or 1.0
    sx = (width * margin_ratio) / max_x
    sy = (height * margin_ratio) / max_y
    return min(sx, sy)


def _map_points(points: Iterable[Point], offset: Tuple[float, float]) -> List[QPointF]:
    dx, dy = offset
    return [QPointF(x + dx, y + dy) for (x, y) in points]


def draw_polyline(
    painter: QPainter,
    points: Sequence[Point],
    *,
    color: str = "#606060",
    width: float = 0.0,
    offset: Tuple[float, float] = (0.0, 0.0),
) -> None:
    """Draw a simple polyline with cosmetic width by default."""

    if len(points) < 2:
        return
    pen = QPen(QColor(color))
    pen.setWidthF(width)
    painter.setPen(pen)
    painter.drawPolyline(_map_points(points, offset))


def draw_track(
    painter: QPainter,
    *,
    centerline: Sequence[Point],
    inner: Sequence[Point] | None = None,
    outer: Sequence[Point] | None = None,
    offset: Tuple[float, float] = (0.0, 0.0),
    center_color: str = "#606060",
    inner_color: str = "#808080",
    outer_color: str = "#808080",
) -> None:
    """Draw the centerline and optional inner/outer edges of a track."""

    draw_polyline(painter, centerline, color=center_color, offset=offset)
    if inner:
        draw_polyline(painter, inner, color=inner_color, offset=offset)
    if outer:
        draw_polyline(painter, outer, color=outer_color, offset=offset)


def draw_teeth_markers(
    painter: QPainter,
    markers: Sequence[Tuple[Point, Point]],
    *,
    color: str = "#444444",
    width: float = 0.0,
    offset: Tuple[float, float] = (0.0, 0.0),
) -> None:
    """Draw tooth markers along a track edge."""

    if not markers:
        return
    pen = QPen(QColor(color))
    pen.setWidthF(width)
    painter.setPen(pen)
    dx, dy = offset
    for a, b in markers:
        painter.drawLine(
            QPointF(a[0] + dx, a[1] + dy),
            QPointF(b[0] + dx, b[1] + dy),
        )


def draw_marker(
    painter: QPainter,
    point: Point,
    *,
    radius: float = 1.5,
    color: str = "#e62739",
    offset: Tuple[float, float] = (0.0, 0.0),
) -> None:
    """Draw a small filled-like marker using a cosmetic pen."""

    pen = QPen(QColor(color))
    pen.setWidthF(0)
    painter.setPen(pen)
    painter.drawEllipse(QPointF(point[0] + offset[0], point[1] + offset[1]), radius, radius)


def draw_wheel(
    painter: QPainter,
    *,
    center: Point,
    radius: float,
    tooth_count: int = 0,
    tooth_index: int = 0,
    contact_angle: float = 0.0,
    roll_sign: float = 1.0,
    tooth_length: float | None = None,
    color: str = "#1f77b4",
    offset: Tuple[float, float] = (0.0, 0.0),
) -> None:
    """Draw a wheel with optional teeth aligned to the contact angle."""

    pen = QPen(QColor(color))
    pen.setWidthF(0)
    painter.setPen(pen)
    painter.drawEllipse(
        QPointF(center[0] + offset[0], center[1] + offset[1]),
        radius,
        radius,
    )

    if tooth_count <= 0:
        return

    effective_tooth_len = tooth_length if (tooth_length is not None) else max(radius * 0.12, 0.8)
    dx, dy = offset

    for k in range(tooth_count):
        angle = contact_angle + roll_sign * 2.0 * math.pi * ((k - tooth_index) / float(tooth_count))
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        inner_r = radius
        outer_r = radius + effective_tooth_len
        painter.drawLine(
            QPointF(center[0] + dx + inner_r * cos_a, center[1] + dy + inner_r * sin_a),
            QPointF(center[0] + dx + outer_r * cos_a, center[1] + dy + outer_r * sin_a),
        )


def build_track_teeth_markers_from_segments(
    segments: Sequence[modular_tracks.TrackSegment],
    *,
    pitch_mm: float,
    half_width: float,
    total_length: float,
    sign_side: float = 1.0,
) -> List[Tuple[Point, Point]]:
    """Compute tooth markers directly from the segment model.

    The markers are aligned using the normal vector returned by
    :func:`modular_tracks._interpolate_on_segments`, ensuring we rely on the
    segment geometry rather than an approximate polyline.
    """

    markers: List[Tuple[Point, Point]] = []
    if not segments or pitch_mm <= 0.0 or total_length <= 0.0:
        return markers

    tooth_len = max(pitch_mm * 0.35, 0.8)
    teeth_count = max(1, int(total_length / pitch_mm))
    safe_length = max(total_length, 1e-9)

    for t_idx in range(teeth_count):
        s = (pitch_mm * t_idx) % safe_length
        C, _, N_vec = modular_tracks._interpolate_on_segments(s, segments)
        nx, ny = N_vec
        x_edge = C[0] + sign_side * nx * half_width
        y_edge = C[1] + sign_side * ny * half_width
        markers.append(
            (
                (x_edge, y_edge),
                (
                    x_edge + sign_side * nx * tooth_len,
                    y_edge + sign_side * ny * tooth_len,
                ),
            )
        )

    return markers
