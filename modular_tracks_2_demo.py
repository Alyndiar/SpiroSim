"""
Démonstration simple pour `modular_tracks_2.py`.

Ce module construit une piste modulaire uniquement avec les fonctions de
`modular_tracks_2` puis anime une roue qui la parcourt. Sont dessinés :
  - les deux côtés de la piste et sa médiane,
  - un indicateur de départ pour la roue,
  - la roue, le point de contact courant, le trou utilisé et le tracé final
    animés pas-à-pas (la roue/contact/trou sont redessinés à chaque étape
    pour apparaître mobiles).

Exécution rapide :
    python modular_tracks_2_demo.py
"""

import math
import sys
from typing import List, Optional, Tuple

from PySide6.QtCore import QPointF, QTimer
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QApplication, QWidget

import modular_tracks_2 as modular_tracks

Point = Tuple[float, float]


def _estimate_track_half_width(segments: List[modular_tracks.TrackSegment]) -> float:
    """Estime la demi-largeur de piste à partir des segments connus."""

    for seg in segments:
        if seg.kind == "arc":
            width = abs(seg.rM - seg.R_track) * 2.0
        elif seg.kind == "line":
            width = abs(seg.R_track) * 2.0
        else:
            continue
        if width > 0:
            return width * 0.5
    return 5.0  # valeur de repli (mm)


def _compute_track_polylines(
    track: modular_tracks.TrackBuildResult,
    samples: int = 400,
    *,
    half_width: Optional[float] = None,
) -> Tuple[List[Point], List[Point], List[Point], float]:
    """Retourne (centre, côté intérieur, côté extérieur, demi-largeur)."""

    segments = track.segments
    effective_half_width = half_width if (half_width and half_width > 0.0) else None
    if effective_half_width is None:
        effective_half_width = _estimate_track_half_width(segments)

    L = track.total_length
    centerline: List[Point] = track.points if track.points else []

    if not centerline:
        for i in range(samples + 1):
            s = (L * i) / float(max(samples, 1))
            C, _, _ = modular_tracks._interpolate_on_segments(s, segments)
            centerline.append(C)

    inner: List[Point] = []
    outer: List[Point] = []

    for i in range(samples + 1):
        s = (L * i) / float(max(samples, 1))
        C, _, N = modular_tracks._interpolate_on_segments(s, segments)
        x, y = C
        nx, ny = N
        inner.append((x - nx * effective_half_width, y - ny * effective_half_width))
        outer.append((x + nx * effective_half_width, y + ny * effective_half_width))

    return centerline, inner, outer, effective_half_width


def _compute_animation_sequences(
    notation: str,
    wheel_teeth: int = 84,
    hole_index: float = 9.5,
    hole_spacing_mm: float = 1.0,
    steps: int = 720,
    relation: str = "dedans",
    wheel_phase_teeth: float = 0.0,
    inner_teeth: int = 96,
    outer_teeth: int = 144,
    pitch_mm_per_tooth: float = 0.65,
) -> Tuple[
    modular_tracks.TrackBuildResult,
    List[Point],
    List[Point],
    List[Point],
    float,
    float,
]:
    """
    Prépare la piste et les positions (stylo, centre de roue, contact).

    Retourne (track, stylo_points, wheel_centers, contact_points, half_width, r_wheel).
    """

    if steps <= 1:
        steps = 2

    track = modular_tracks.build_track_from_notation(
        notation,
        inner_teeth=inner_teeth,
        outer_teeth=outer_teeth,
        pitch_mm_per_tooth=pitch_mm_per_tooth,
    )
    segments = track.segments
    if not segments:
        return track, [], [], [], 0.0, 0.0

    width_mm = (
        (float(outer_teeth) - float(inner_teeth))
        * pitch_mm_per_tooth
        / (2.0 * math.pi)
        if outer_teeth and inner_teeth and outer_teeth > inner_teeth
        else 0.0
    )
    half_width = width_mm * 0.5 if width_mm > 0.0 else _estimate_track_half_width(segments)
    track_offset_teeth = float(track.offset_teeth or 0.0)

    centerline, _, _, half_width = _compute_track_polylines(
        track, half_width=half_width
    )

    # Rayon de la roue et distance du trou par rapport au centre
    r_wheel = (wheel_teeth * pitch_mm_per_tooth) / (2.0 * math.pi)
    d = r_wheel - hole_index * hole_spacing_mm
    if d < 0.0:
        d = 0.0

    # Longueur parcourue totale
    L = track.total_length
    if L <= 0:
        return track, [], [], [], half_width, r_wheel

    if track.total_teeth > 0:
        N_track = max(1, int(round(track.total_teeth)))
    else:
        N_track = wheel_teeth
    N_w = max(1, int(wheel_teeth))
    g = math.gcd(N_track, N_w)
    if g <= 0:
        g = 1
    nb_laps = N_w // g if N_w >= g else 1
    if nb_laps < 1:
        nb_laps = 1
    s_max = L * float(nb_laps)

    stylo_points: List[Point] = []
    wheel_centers: List[Point] = []
    contact_points: List[Point] = []

    sign_side = -1.0 if relation == "dedans" else 1.0

    for i in range(steps):
        s = s_max * i / float(steps - 1)
        C, _, N_vec = modular_tracks._interpolate_on_segments(s % L, segments)
        x_track, y_track = C
        nx, ny = N_vec

        contact_offset = sign_side * half_width
        contact_x = x_track + contact_offset * nx
        contact_y = y_track + contact_offset * ny
        contact_points.append((contact_x, contact_y))

        center_offset = contact_offset + sign_side * r_wheel
        cx = x_track + center_offset * nx
        cy = y_track + center_offset * ny
        wheel_centers.append((cx, cy))

        angle_contact = math.atan2(contact_y - cy, contact_x - cx)
        teeth_rolled = (s / pitch_mm_per_tooth) - float(wheel_phase_teeth) + track_offset_teeth
        phi = angle_contact + 2.0 * math.pi * (teeth_rolled / float(N_w))
        px = cx + d * math.cos(phi)
        py = cy + d * math.sin(phi)
        stylo_points.append((px, py))

    # Remplacer track.points par la médiane recalculée pour l'affichage
    track.points = centerline

    return track, stylo_points, wheel_centers, contact_points, half_width, r_wheel


class ModularTrackDemo(QWidget):
    """Widget simple qui anime le tracé d'une piste modulaire."""

    def __init__(
        self,
        parent=None,
        *,
        auto_start: bool = True,
        notation: str = "-18-C+D+B-C+D+",
        wheel_teeth: int = 84,
        hole_index: float = 9.5,
        hole_spacing: float = 1.0,
        relation: str = "dedans",
        steps: int = 720,
        wheel_phase_teeth: float = 0.0,
        inner_teeth: int = 96,
        outer_teeth: int = 144,
        pitch_mm_per_tooth: float = 0.65,
        scale: float = 1.0,
    ):
        super().__init__(parent)
        self.setWindowTitle("Démo modular_tracks_2")

        self._interval_ms = 20
        self._speed_pts_per_s: float = 1.0
        self._progress: float = 0.0
        self._full_path = False

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._on_tick)

        self.track = modular_tracks.TrackBuildResult(
            points=[], total_length=0.0, total_teeth=0.0, offset_teeth=0, segments=[]
        )
        self.stylo_points: List[Point] = []
        self.wheel_centers: List[Point] = []
        self.contact_points: List[Point] = []
        self.inner_side: List[Point] = []
        self.outer_side: List[Point] = []
        self.half_width = 0.0
        self.r_wheel = 0.0
        self.current_step = 0

        self.set_configuration(
            notation=notation,
            wheel_teeth=wheel_teeth,
            hole_index=hole_index,
            hole_spacing=hole_spacing,
            relation=relation,
            steps=steps,
            wheel_phase_teeth=wheel_phase_teeth,
            inner_teeth=inner_teeth,
            outer_teeth=outer_teeth,
            pitch_mm_per_tooth=pitch_mm_per_tooth,
            scale=scale,
        )

        if auto_start:
            self.start_animation()

    def set_configuration(
        self,
        *,
        notation: str,
        wheel_teeth: int,
        hole_index: float,
        hole_spacing: float,
        relation: str,
        steps: int,
        wheel_phase_teeth: float,
        inner_teeth: int,
        outer_teeth: int,
        pitch_mm_per_tooth: float,
        scale: float = 1.0,
    ):
        (
            track,
            stylo_points,
            wheel_centers,
            contact_points,
            half_width,
            r_wheel,
        ) = _compute_animation_sequences(
            notation,
            wheel_teeth=wheel_teeth,
            hole_index=hole_index,
            hole_spacing_mm=hole_spacing,
            steps=steps,
            relation=relation,
            wheel_phase_teeth=wheel_phase_teeth,
            inner_teeth=inner_teeth,
            outer_teeth=outer_teeth,
            pitch_mm_per_tooth=pitch_mm_per_tooth,
        )

        if not track.segments:
            self.track = track
            self.track.points = []
            self.stylo_points = []
            self.wheel_centers = []
            self.contact_points = []
            self.inner_side = []
            self.outer_side = []
            self.half_width = 0.0
            self.r_wheel = 0.0
            self.current_step = 0
            self._progress = 0.0
            self._full_path = False
            self._update_viewport()
            self.update()
            return

        def _scale_pts(pts: List[Point]) -> List[Point]:
            if scale == 1.0:
                return pts
            return [(x * scale, y * scale) for (x, y) in pts]

        self.track = track
        self.track.points = _scale_pts(track.points or [])
        self.stylo_points = _scale_pts(stylo_points)
        self.wheel_centers = _scale_pts(wheel_centers)
        self.contact_points = _scale_pts(contact_points)
        self.half_width = half_width * scale
        self.r_wheel = r_wheel * scale
        self.current_step = 0
        self._progress = 0.0
        self._full_path = False

        centerline, inner, outer, _ = _compute_track_polylines(track)
        self.track.points = _scale_pts(centerline)
        self.inner_side = _scale_pts(inner)
        self.outer_side = _scale_pts(outer)

        self._update_viewport()
        self.update()

    def set_speed(self, points_per_second: float):
        self._speed_pts_per_s = max(0.0, float(points_per_second))
        self._full_path = self._speed_pts_per_s == 0.0
        if self._full_path:
            self.stop_animation()
            if self.stylo_points:
                self.current_step = len(self.stylo_points) - 1
                self.update()

    def start_animation(self):
        if not self.stylo_points or self._full_path:
            return
        if not self.timer.isActive():
            self.timer.start(self._interval_ms)

    def stop_animation(self):
        self.timer.stop()

    def reset_animation(self):
        self._progress = 0.0
        self._full_path = self._speed_pts_per_s == 0.0
        if self._full_path and self.stylo_points:
            self.current_step = len(self.stylo_points) - 1
        else:
            self.current_step = 0
        self.update()

    def _on_tick(self):
        if not self.stylo_points or self._full_path:
            return
        dt = self._interval_ms / 1000.0
        self._progress = (self._progress + self._speed_pts_per_s * dt) % float(
            len(self.stylo_points)
        )
        self.current_step = int(self._progress) % len(self.stylo_points)
        self.update()

    def _update_viewport(self):
        all_points: List[Point] = []
        all_points.extend(self.track.points)
        all_points.extend(self.stylo_points)
        all_points.extend(self.wheel_centers)
        all_points.extend(self.contact_points)
        if not all_points:
            self._scale = 1.0
            self._offset = (0.0, 0.0)
            return

        xs = [p[0] for p in all_points]
        ys = [p[1] for p in all_points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        width = max(max_x - min_x, 1e-6)
        height = max(max_y - min_y, 1e-6)

        margin = 10.0
        min_x -= margin
        max_x += margin
        min_y -= margin
        max_y += margin

        self._offset = (-(min_x + max_x) * 0.5, -(min_y + max_y) * 0.5)
        self._base_width = width + 2.0 * margin
        self._base_height = height + 2.0 * margin

    def resizeEvent(self, event):  # noqa: N802 - signature imposée par Qt
        super().resizeEvent(event)
        self._update_scale()

    def _update_scale(self):
        if getattr(self, "_base_width", 0) <= 0 or getattr(self, "_base_height", 0) <= 0:
            self._scale = 1.0
            return
        sx = self.width() / float(self._base_width)
        sy = self.height() / float(self._base_height)
        self._scale = 0.9 * min(sx, sy)

    def _map_points(self, pts: List[Point]) -> List[QPointF]:
        dx, dy = getattr(self, "_offset", (0.0, 0.0))
        return [QPointF((x + dx), (y + dy)) for (x, y) in pts]

    def _draw_polyline(self, painter: QPainter, pts: List[Point]):
        if len(pts) < 2:
            return
        painter.drawPolyline(self._map_points(pts))

    def paintEvent(self, event):  # noqa: N802 - signature imposée par Qt
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        self._update_scale()
        painter.translate(self.width() * 0.5, self.height() * 0.5)
        painter.scale(self._scale, -self._scale)

        # Piste : côtés et médiane
        painter.setPen(QPen(QColor("#888"), 0))
        self._draw_polyline(painter, self.track.points)

        painter.setPen(QPen(QColor("#555"), 0))
        self._draw_polyline(painter, self.inner_side)
        self._draw_polyline(painter, self.outer_side)

        # Indicateur de départ (point de contact à s=0)
        if self.contact_points:
            start = self.contact_points[0]
            painter.setPen(QPen(QColor("#00aa00"), 0))
            painter.drawEllipse(QPointF(start[0] + self._offset[0], start[1] + self._offset[1]), 1.5, 1.5)

        if not self.stylo_points:
            painter.end()
            return

        # Tracé final déjà dessiné
        painter.setPen(QPen(QColor("#cc3366"), 0))
        self._draw_polyline(painter, self.stylo_points[: self.current_step + 1])

        idx = self.current_step
        wheel_center = self.wheel_centers[idx]
        contact = self.contact_points[idx]
        hole = self.stylo_points[idx]

        # Roue
        painter.setPen(QPen(QColor("#1f77b4"), 0))
        painter.drawEllipse(
            QPointF(wheel_center[0] + self._offset[0], wheel_center[1] + self._offset[1]),
            self.r_wheel,
            self.r_wheel,
        )

        # Point de contact
        painter.setPen(QPen(QColor("#ff9900"), 0))
        painter.drawEllipse(QPointF(contact[0] + self._offset[0], contact[1] + self._offset[1]), 1.3, 1.3)

        # Trou / stylo courant
        painter.setPen(QPen(QColor("#e62739"), 0))
        painter.drawEllipse(QPointF(hole[0] + self._offset[0], hole[1] + self._offset[1]), 1.5, 1.5)

        painter.end()


def main():
    app = QApplication(sys.argv)
    demo = ModularTrackDemo()
    demo.resize(800, 600)
    demo.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
