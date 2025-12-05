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

from PySide6.QtCore import QTimer
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QApplication, QWidget

import modular_tracks_2 as modular_tracks
import drawing

Point = Tuple[float, float]


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

    centerline, _, _, half_width = modular_tracks.compute_track_polylines(
        track, half_width=bundle.context.half_width
    )

    track_teeth_markers = drawing.build_track_teeth_markers_from_segments(
        track.segments,
        pitch_mm=bundle.context.pitch_mm_per_tooth,
        half_width=bundle.context.half_width,
        total_length=bundle.context.track_length,
        sign_side=bundle.context.sign_side,
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

        centerline, inner, outer, _ = modular_tracks.compute_track_polylines(
            track, half_width=half_width
        )
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

    def paintEvent(self, event):  # noqa: N802 - signature imposée par Qt
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        self._update_scale()
        painter.translate(self.width() * 0.5, self.height() * 0.5)
        painter.scale(self._scale, -self._scale)

        drawing.draw_track(
            painter,
            centerline=self.track.points,
            inner=self.inner_side,
            outer=self.outer_side,
            offset=self._offset,
            center_color="#888",
            inner_color="#555",
            outer_color="#555",
        )

        drawing.draw_teeth_markers(
            painter,
            self.track_teeth_markers,
            color="#444",
            offset=self._offset,
        )

        # Indicateur de départ (point de contact à s=0)
        if self.contact_points:
            start = self.contact_points[0]
            drawing.draw_marker(
                painter,
                start,
                radius=1.5,
                color="#00aa00",
                offset=self._offset,
            )

        if not self.stylo_points:
            painter.end()
            return

        # Tracé final déjà dessiné
        drawing.draw_polyline(
            painter,
            self.stylo_points[: self.current_step + 1],
            color="#cc3366",
            offset=self._offset,
        )

        idx = self.current_step
        wheel_center = self.wheel_centers[idx]
        contact = self.contact_points[idx]
        hole = self.stylo_points[idx]
        marker0 = self.markers_angle0[idx]

        wheel_idx = self.wheel_tooth_indices[idx] if idx < len(self.wheel_tooth_indices) else 0
        track_idx = self.track_tooth_indices[idx] if idx < len(self.track_tooth_indices) else 0
        angle_contact = math.atan2(contact[1] - wheel_center[1], contact[0] - wheel_center[0])

        # Roue
        drawing.draw_wheel(
            painter,
            center=wheel_center,
            radius=self.r_wheel,
            tooth_count=self.wheel_teeth_count,
            tooth_index=wheel_idx,
            contact_angle=angle_contact,
            roll_sign=self.roll_sign,
            offset=self._offset,
        )

        # Marqueur d'angle 0 sur le bord de la roue (permet de suivre la rotation).
        drawing.draw_marker(
            painter,
            marker0,
            radius=1.2,
            color="#000000",
            offset=self._offset,
        )

        # Point de contact
        drawing.draw_marker(
            painter,
            contact,
            radius=1.3,
            color="#ff9900",
            offset=self._offset,
        )

        # Trou / stylo courant
        drawing.draw_marker(
            painter,
            hole,
            radius=1.5,
            color="#e62739",
            offset=self._offset,
        )

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
