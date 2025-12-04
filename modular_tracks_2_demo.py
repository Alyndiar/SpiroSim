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
    List[Point],
    float,
    float,
    List[Tuple[Point, Point]],
    int,
    int,
    List[int],
    List[int],
    float,
]:
    """
    Prépare la piste, les positions (stylo, centre, contact) et les métadonnées
    de dents pour l'affichage animé.
    """

    track = modular_tracks.build_track_from_notation(
        notation,
        inner_teeth=inner_teeth,
        outer_teeth=outer_teeth,
        pitch_mm_per_tooth=pitch_mm_per_tooth,
    )
    if not track.segments:
        return track, [], [], [], [], 0.0, 0.0, [], 0, 0, [], [], 0.0

    bundle = modular_tracks._generate_track_roll_bundle(
        track=track,
        notation=notation,
        wheel_teeth=wheel_teeth,
        hole_index=hole_index,
        hole_spacing_mm=hole_spacing_mm,
        steps=steps,
        relation=relation,
        wheel_phase_teeth=wheel_phase_teeth,
        inner_teeth=inner_teeth,
        outer_teeth=outer_teeth,
        pitch_mm_per_tooth=pitch_mm_per_tooth,
    )

    centerline, _, _, half_width = _compute_track_polylines(
        track, half_width=bundle.context.half_width
    )

    track_teeth_markers: List[Tuple[Point, Point]] = []
    tooth_len = max(bundle.context.pitch_mm_per_tooth * 0.35, 0.8)
    L = bundle.context.track_length
    for t_idx in range(bundle.context.N_track):
        s = (bundle.context.pitch_mm_per_tooth * t_idx) % max(L, 1e-9)
        C, _, N_vec = modular_tracks._interpolate_on_segments(s, track.segments)
        nx, ny = N_vec
        x_edge = C[0] + bundle.context.sign_side * nx * bundle.context.half_width
        y_edge = C[1] + bundle.context.sign_side * ny * bundle.context.half_width
        track_teeth_markers.append(
            (
                (x_edge, y_edge),
                (
                    x_edge + bundle.context.sign_side * nx * tooth_len,
                    y_edge + bundle.context.sign_side * ny * tooth_len,
                ),
            )
        )

    track.points = centerline

    return (
        track,
        bundle.stylo,
        bundle.centre,
        bundle.contact,
        bundle.marker0,
        half_width,
        bundle.context.r_wheel,
        track_teeth_markers,
        bundle.context.N_wheel,
        bundle.context.N_track,
        bundle.wheel_teeth_indices,
        bundle.track_teeth_indices,
        -bundle.context.sign_side,
    )


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
        self.markers_angle0: List[Point] = []
        self.inner_side: List[Point] = []
        self.outer_side: List[Point] = []
        self.track_teeth_markers: List[Tuple[Point, Point]] = []
        self.half_width = 0.0
        self.r_wheel = 0.0
        self.wheel_teeth_count = 0
        self.track_teeth_count = 0
        self.wheel_tooth_indices: List[int] = []
        self.track_tooth_indices: List[int] = []
        self.roll_sign = 0.0
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
            markers_angle0,
            half_width,
            r_wheel,
            track_teeth_markers,
            wheel_teeth_count,
            track_teeth_count,
            wheel_tooth_indices,
            track_tooth_indices,
            roll_sign,
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
            self.markers_angle0 = []
            self.inner_side = []
            self.outer_side = []
            self.track_teeth_markers = []
            self.half_width = 0.0
            self.r_wheel = 0.0
            self.wheel_teeth_count = 0
            self.track_teeth_count = 0
            self.wheel_tooth_indices = []
            self.track_tooth_indices = []
            self.roll_sign = 0.0
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
        self.markers_angle0 = _scale_pts(markers_angle0)
        self.track_teeth_markers = [
            (_scale_pts([a])[0], _scale_pts([b])[0]) for (a, b) in track_teeth_markers
        ]
        self.half_width = half_width * scale
        self.r_wheel = r_wheel * scale
        self.wheel_teeth_count = wheel_teeth_count
        self.track_teeth_count = track_teeth_count
        self.wheel_tooth_indices = wheel_tooth_indices
        self.track_tooth_indices = track_tooth_indices
        self.roll_sign = roll_sign
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
        all_points.extend(self.markers_angle0)
        for a, b in self.track_teeth_markers:
            all_points.append(a)
            all_points.append(b)
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

        painter.setPen(QPen(QColor("#444"), 0))
        for a, b in self.track_teeth_markers:
            painter.drawLine(
                QPointF(a[0] + self._offset[0], a[1] + self._offset[1]),
                QPointF(b[0] + self._offset[0], b[1] + self._offset[1]),
            )

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
        marker0 = self.markers_angle0[idx]

        wheel_idx = self.wheel_tooth_indices[idx] if idx < len(self.wheel_tooth_indices) else 0
        track_idx = self.track_tooth_indices[idx] if idx < len(self.track_tooth_indices) else 0
        angle_contact = math.atan2(contact[1] - wheel_center[1], contact[0] - wheel_center[0])

        # Roue
        painter.setPen(QPen(QColor("#1f77b4"), 0))
        painter.drawEllipse(
            QPointF(wheel_center[0] + self._offset[0], wheel_center[1] + self._offset[1]),
            self.r_wheel,
            self.r_wheel,
        )

        if self.wheel_teeth_count > 0:
            tooth_len = max(self.r_wheel * 0.12, 0.8)
            for k in range(self.wheel_teeth_count):
                angle = angle_contact + self.roll_sign * 2.0 * math.pi * (
                    (k - wheel_idx) / float(self.wheel_teeth_count)
                )
                cos_a, sin_a = math.cos(angle), math.sin(angle)
                inner_r = self.r_wheel
                outer_r = self.r_wheel + tooth_len
                painter.drawLine(
                    QPointF(
                        wheel_center[0] + self._offset[0] + inner_r * cos_a,
                        wheel_center[1] + self._offset[1] + inner_r * sin_a,
                    ),
                    QPointF(
                        wheel_center[0] + self._offset[0] + outer_r * cos_a,
                        wheel_center[1] + self._offset[1] + outer_r * sin_a,
                    ),
                )

        # Marqueur d'angle 0 sur le bord de la roue (permet de suivre la rotation).
        painter.setPen(QPen(QColor("#000000"), 0))
        painter.drawEllipse(
            QPointF(marker0[0] + self._offset[0], marker0[1] + self._offset[1]),
            1.2,
            1.2,
        )

        # Point de contact
        painter.setPen(QPen(QColor("#ff9900"), 0))
        painter.drawEllipse(QPointF(contact[0] + self._offset[0], contact[1] + self._offset[1]), 1.3, 1.3)

        # Trou / stylo courant
        painter.setPen(QPen(QColor("#e62739"), 0))
        painter.drawEllipse(QPointF(hole[0] + self._offset[0], hole[1] + self._offset[1]), 1.5, 1.5)

        painter.resetTransform()
        bg_color = self.palette().color(self.backgroundRole())
        avg_rgb = (bg_color.red() + bg_color.green() + bg_color.blue()) / 3.0
        text_color = QColor("#000000" if avg_rgb > 128 else "#ffffff")
        painter.setPen(QPen(text_color))
        painter.drawText(
            10,
            20,
            f"Contact dents: roue {wheel_idx}/{max(1, self.wheel_teeth_count)} - piste {track_idx}/{max(1, self.track_teeth_count)}",
        )
        painter.drawText(
            10,
            40,
            f"Dents roue: {self.wheel_teeth_count} | Dents piste: {self.track_teeth_count}",
        )

        painter.end()


def main():
    app = QApplication(sys.argv)
    demo = ModularTrackDemo()
    demo.resize(800, 600)
    demo.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
