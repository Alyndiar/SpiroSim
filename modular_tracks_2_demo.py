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
from typing import List, Tuple

from PySide6.QtCore import QPointF, QTimer
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QApplication, QWidget

import modular_tracks_2 as modular_tracks

Point = Tuple[float, float]


def _compute_animation_sequences(
    notation: str,
    wheel_size: int = 84,
    hole_offset: float = 9.5,
    steps: int = 720,
    relation: str = "dedans",
    phase_offset: float = 0.0,
    inner_size: int = 96,
    outer_size: int = 144,
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
    de marqueurs pour l'affichage animé.
    """

    track, bundle = modular_tracks.build_track_and_bundle_from_notation(
        notation=notation,
        wheel_size=wheel_size,
        hole_offset=hole_offset,
        steps=steps,
        relation=relation,
        phase_offset=phase_offset,
        inner_size=inner_size,
        outer_size=outer_size,
    )
    if not track.segments:
        return track, [], [], [], [], 0.0, 0.0, [], 0, 0, [], [], 0.0

    centerline, _, _, half_width = modular_tracks.compute_track_polylines(
        track, samples=800, half_width=track.half_width
    )

    track_markers: List[Tuple[Point, Point]] = []
    tick_len = max(bundle.context.r_wheel * 0.12, 0.8)
    L = bundle.context.track_length
    for t_idx in range(bundle.context.track_size):
        s = float(t_idx) % max(L, 1e-9)
        C, _, N_vec = modular_tracks._interpolate_on_segments(s, track.segments)
        nx, ny = N_vec
        x_edge = C[0] + bundle.context.sign_side * nx * track.half_width
        y_edge = C[1] + bundle.context.sign_side * ny * track.half_width
        track_markers.append(
            (
                (x_edge, y_edge),
                (
                    x_edge + bundle.context.sign_side * nx * tick_len,
                    y_edge + bundle.context.sign_side * ny * tick_len,
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
        track_markers,
        bundle.context.wheel_size,
        bundle.context.track_size,
        bundle.wheel_marker_indices,
        bundle.track_marker_indices,
        -bundle.context.sign_side,
    )


class ModularTrackDemo(QWidget):
    """Widget simple qui anime le tracé d'une piste modulaire."""

    def __init__(
        self,
        parent=None,
        *,
        auto_start: bool = True,
        notation: str = "+A60+L144+A60+L144",
        wheel_size: int = 84,
        hole_offset: float = 9.5,
        relation: str = "dedans",
        steps: int = 720,
        phase_offset: float = 0.0,
        inner_size: int = 96,
        outer_size: int = 144,
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
            segments=[],
            points=[],
            width=0.0,
            half_width=0.0,
            inner_length=0.0,
            outer_length=0.0,
            origin_offset=0.0,
            origin_angle_offset=0.0,
            ring=modular_tracks.ReferenceRing(),
        )
        self.stylo_points: List[Point] = []
        self.wheel_centers: List[Point] = []
        self.contact_points: List[Point] = []
        self.markers_angle0: List[Point] = []
        self.inner_side: List[Point] = []
        self.outer_side: List[Point] = []
        self.track_markers: List[Tuple[Point, Point]] = []
        self.half_width = 0.0
        self.r_wheel = 0.0
        self.wheel_size_count = 0
        self.track_size_count = 0
        self.wheel_marker_indices: List[int] = []
        self.track_marker_indices: List[int] = []
        self.roll_sign = 0.0
        self.current_step = 0
        self.wheel_orientations: List[Tuple[float, float]] = []
        self.wheel_shape_local: List[Point] = []
        self.wheel_marker_local: Point | None = None

        self.set_configuration(
            notation=notation,
            wheel_size=wheel_size,
            hole_offset=hole_offset,
            relation=relation,
            steps=steps,
            phase_offset=phase_offset,
            inner_size=inner_size,
            outer_size=outer_size,
            scale=scale,
        )

        if auto_start:
            self.start_animation()

    def set_configuration(
        self,
        *,
        notation: str,
        wheel_size: int,
        hole_offset: float,
        relation: str,
        steps: int,
        phase_offset: float,
        inner_size: int,
        outer_size: int,
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
            track_markers,
            wheel_size_count,
            track_size_count,
            wheel_marker_indices,
            track_marker_indices,
            roll_sign,
        ) = _compute_animation_sequences(
            notation,
            wheel_size=wheel_size,
            hole_offset=hole_offset,
            steps=steps,
            relation=relation,
            phase_offset=phase_offset,
            inner_size=inner_size,
            outer_size=outer_size,
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
            self.track_markers = []
            self.half_width = 0.0
            self.r_wheel = 0.0
            self.wheel_size_count = 0
            self.track_size_count = 0
            self.wheel_marker_indices = []
            self.track_marker_indices = []
            self.roll_sign = 0.0
            self.current_step = 0
            self.wheel_orientations = []
            self.wheel_shape_local = []
            self.wheel_marker_local = None
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
        self.track_markers = [
            (_scale_pts([a])[0], _scale_pts([b])[0]) for (a, b) in track_markers
        ]
        self.half_width = half_width * scale
        self.r_wheel = r_wheel * scale
        self.wheel_size_count = wheel_size_count
        self.track_size_count = track_size_count
        self.wheel_marker_indices = wheel_marker_indices
        self.track_marker_indices = track_marker_indices
        self.roll_sign = roll_sign
        self.current_step = 0
        self.wheel_orientations = []
        self.wheel_shape_local = []
        self.wheel_marker_local = None
        self._progress = 0.0
        self._full_path = False

        centerline, inner, outer, _ = modular_tracks.compute_track_polylines(
            track, samples=800, half_width=half_width
        )
        self.track.points = _scale_pts(centerline)
        self.inner_side = _scale_pts(inner)
        self.outer_side = _scale_pts(outer)

        self._update_viewport()
        self.update()

    def set_debug_sequences(
        self,
        *,
        track: modular_tracks.TrackBuildResult,
        stylo_points: List[Point],
        wheel_centers: List[Point],
        contact_points: List[Point],
        markers_angle0: List[Point],
        track_markers: List[Tuple[Point, Point]],
        r_wheel: float,
        wheel_size_count: int,
        track_size_count: int,
        wheel_marker_indices: List[int],
        track_marker_indices: List[int],
        roll_sign: float,
        wheel_orientations: List[Tuple[float, float]],
        wheel_shape_local: List[Point],
        wheel_marker_local: Point | None,
        scale: float = 1.0,
    ):
        def _scale_pts(pts: List[Point]) -> List[Point]:
            if scale == 1.0:
                return pts
            return [(x * scale, y * scale) for (x, y) in pts]

        if not track.segments:
            self.track = track
            self.track.points = []
            self.stylo_points = []
            self.wheel_centers = []
            self.contact_points = []
            self.markers_angle0 = []
            self.inner_side = []
            self.outer_side = []
            self.track_markers = []
            self.half_width = 0.0
            self.r_wheel = 0.0
            self.wheel_size_count = 0
            self.track_size_count = 0
            self.wheel_marker_indices = []
            self.track_marker_indices = []
            self.roll_sign = 0.0
            self.current_step = 0
            self.wheel_orientations = []
            self.wheel_shape_local = []
            self.wheel_marker_local = None
            self._progress = 0.0
            self._full_path = False
            self._update_viewport()
            self.update()
            return

        self.track = track
        self.track.points = _scale_pts(track.points or [])
        self.stylo_points = _scale_pts(stylo_points)
        self.wheel_centers = _scale_pts(wheel_centers)
        self.contact_points = _scale_pts(contact_points)
        self.markers_angle0 = _scale_pts(markers_angle0)
        self.track_markers = [
            (_scale_pts([a])[0], _scale_pts([b])[0]) for (a, b) in track_markers
        ]
        self.half_width = track.half_width * scale
        self.r_wheel = r_wheel * scale
        self.wheel_size_count = wheel_size_count
        self.track_size_count = track_size_count
        self.wheel_marker_indices = wheel_marker_indices
        self.track_marker_indices = track_marker_indices
        self.roll_sign = roll_sign
        self.current_step = 0
        self.wheel_orientations = wheel_orientations
        self.wheel_shape_local = _scale_pts(wheel_shape_local)
        self.wheel_marker_local = _scale_pts([wheel_marker_local])[0] if wheel_marker_local else None
        self._progress = 0.0
        self._full_path = False

        centerline, inner, outer, _ = modular_tracks.compute_track_polylines(
            track, samples=800, half_width=track.half_width
        )
        self.track.points = _scale_pts(centerline)
        self.inner_side = _scale_pts(inner)
        self.outer_side = _scale_pts(outer)

        if self.wheel_centers and self.wheel_shape_local:
            self.r_wheel = max(
                self.r_wheel,
                max(math.hypot(x, y) for x, y in self.wheel_shape_local) if self.wheel_shape_local else 0.0,
            )

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
        for a, b in self.track_markers:
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

    def _draw_transformed_shape(
        self,
        painter: QPainter,
        points: List[Point],
        center: Point,
        cos_a: float,
        sin_a: float,
    ):
        if len(points) < 2:
            return
        dx, dy = getattr(self, "_offset", (0.0, 0.0))
        mapped = [
            QPointF(
                center[0] + dx + (x * cos_a - y * sin_a),
                center[1] + dy + (x * sin_a + y * cos_a),
            )
            for (x, y) in points
        ]
        painter.drawPolyline(mapped)

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
        for a, b in self.track_markers:
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

        wheel_idx = self.wheel_marker_indices[idx] if idx < len(self.wheel_marker_indices) else 0
        track_idx = self.track_marker_indices[idx] if idx < len(self.track_marker_indices) else 0
        angle_contact = math.atan2(contact[1] - wheel_center[1], contact[0] - wheel_center[0])

        # Roue
        painter.setPen(QPen(QColor("#1f77b4"), 0))
        if self.wheel_shape_local and idx < len(self.wheel_orientations):
            cos_a, sin_a = self.wheel_orientations[idx]
            self._draw_transformed_shape(painter, self.wheel_shape_local, wheel_center, cos_a, sin_a)
        else:
            painter.drawEllipse(
                QPointF(wheel_center[0] + self._offset[0], wheel_center[1] + self._offset[1]),
                self.r_wheel,
                self.r_wheel,
            )

        if self.wheel_size_count > 0:
            tick_len = max(self.r_wheel * 0.12, 0.8)
            for k in range(self.wheel_size_count):
                angle = angle_contact + self.roll_sign * 2.0 * math.pi * (
                    (k - wheel_idx) / float(self.wheel_size_count)
                )
                cos_a, sin_a = math.cos(angle), math.sin(angle)
                inner_r = self.r_wheel
                outer_r = self.r_wheel + tick_len
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
        if self.wheel_marker_local and idx < len(self.wheel_orientations):
            cos_a, sin_a = self.wheel_orientations[idx]
            mx = wheel_center[0] + (self.wheel_marker_local[0] * cos_a - self.wheel_marker_local[1] * sin_a)
            my = wheel_center[1] + (self.wheel_marker_local[0] * sin_a + self.wheel_marker_local[1] * cos_a)
            painter.drawEllipse(QPointF(mx + self._offset[0], my + self._offset[1]), 1.2, 1.2)
        else:
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
        painter.drawEllipse(QPointF(hole[0] + self._offset[0], hole[1] + self._offset[1]), 1.0, 1.0)

        painter.resetTransform()
        bg_color = self.palette().color(self.backgroundRole())
        avg_rgb = (bg_color.red() + bg_color.green() + bg_color.blue()) / 3.0
        text_color = QColor("#000000" if avg_rgb > 128 else "#ffffff")
        painter.setPen(QPen(text_color))
        painter.drawText(
            10,
            20,
            f"Contact repères: roue {wheel_idx}/{max(1, self.wheel_size_count)} - piste {track_idx}/{max(1, self.track_size_count)}",
        )
        painter.drawText(
            10,
            40,
            f"Repères roue: {self.wheel_size_count} | Repères piste: {self.track_size_count}",
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
