from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import List, Optional, Tuple

import modular_tracks
from shape_geometry import (
    ArcSegment,
    BaseCurve,
    CircleCurve,
    EllipseCurve,
    LineSegment,
    ModularTrackCurve,
    build_circle,
    build_drop,
    build_oblong,
    build_rounded_polygon,
    pen_position,
    roll_pen_position,
    wheel_orientation,
    wheel_pen_local_vector,
)
from shape_rsdl import RsdlParseError, parse_analytic_expression

Point = Tuple[float, float]


@dataclass
class GearConfig:
    name: str = "Engrenage"
    gear_type: str = "anneau"   # anneau, roue, rsdl, modulaire
    size: int = 96              # taille de la roue / taille intérieure de l'anneau
    outer_size: int = 144       # anneau : taille extérieure / anneau modulaire
    relation: str = "stationnaire"  # stationnaire / dedans / dehors
    modular_notation: Optional[str] = None  # notation de piste si gear_type == "modulaire"
    rsdl_expression: Optional[str] = None  # expression RSDL si gear_type == "rsdl"


@dataclass
class PathConfig:
    name: str = "Tracé"
    enable: bool = True
    hole_offset: float = 1.0
    hole_direction: float = 0.0
    phase_offset: float = 0.0
    color: str = "blue"            # chaîne telle que saisie / affichée
    color_norm: Optional[str] = None  # valeur normalisée (#rrggbb) pour le dessin
    stroke_width: float = 1.2
    zoom: float = 1.0
    translate_x: float = 0.0
    translate_y: float = 0.0
    rotate_deg: float = 0.0


@dataclass
class LayerConfig:
    name: str = "Couche"
    enable: bool = True
    zoom: float = 1.0                         # zoom de la couche
    translate_x: float = 0.0
    translate_y: float = 0.0
    rotate_deg: float = 0.0
    gears: List[GearConfig] = field(default_factory=list)  # 2 ou 3 engrenages
    paths: List[PathConfig] = field(default_factory=list)


def radius_from_size(size: int) -> float:
    """
    Calcule le rayon d’un cercle de pas ayant une taille donnée.
    """
    if size <= 0:
        return 0.0
    return float(size) / (2.0 * math.pi)


def contact_size_for_relation(gear: GearConfig, relation: str) -> int:
    """
    Taille utilisée pour le contact, selon la relation.
    - anneau + 'dedans' : on utilise la taille intérieure (gear.size)
    - anneau + 'dehors' : on utilise la taille extérieure (gear.outer_size)
    - roue / autres : gear.size
    """
    if gear.gear_type == "anneau":
        if relation == "dehors":
            return gear.outer_size or gear.size
        else:
            return gear.size
    return gear.size


def phase_offset_turns(offset: float, size: int) -> float:
    """
    Convertit un décalage en unités (O) en fraction de tour (O/S).
    """
    if size <= 0:
        return 0.0
    return float(offset) / float(size)


def contact_radius_for_relation(gear: GearConfig, relation: str) -> float:
    """
    Rayon de contact pour un engrenage donné, selon la relation.
    """
    size = contact_size_for_relation(gear, relation)
    return radius_from_size(size)


def _curve_from_analytic_spec(spec, relation: str) -> BaseCurve:
    if spec.__class__.__name__ == "CircleSpec":
        return build_circle(spec.perimeter)
    if spec.__class__.__name__ == "RingSpec":
        if relation == "dehors":
            return build_circle(spec.outer)
        return build_circle(spec.inner)
    if spec.__class__.__name__ == "PolygonSpec":
        return build_rounded_polygon(spec.perimeter, spec.sides, spec.side_size, spec.corner_size)
    if spec.__class__.__name__ == "DropSpec":
        return build_drop(spec.perimeter, spec.opposite, spec.half, spec.link)
    if spec.__class__.__name__ == "OblongSpec":
        return build_oblong(spec.perimeter, spec.cap_size)
    if spec.__class__.__name__ == "EllipseSpec":
        return EllipseCurve(spec.perimeter, spec.axis_a, spec.axis_b)
    return build_circle(1.0)


def _curve_from_gear(gear: GearConfig, relation: str) -> Optional[BaseCurve]:
    if gear.gear_type == "modulaire" and gear.modular_notation:
        track = modular_tracks.build_track_from_notation(
            gear.modular_notation,
            inner_size=gear.size or 1,
            outer_size=gear.outer_size or (gear.size + 1),
            steps_per_unit=3,
        )
        segments = []
        for seg in track.segments:
            if seg.kind == "line":
                segments.append(LineSegment(seg.start, seg.end))
            elif seg.kind == "arc" and seg.center is not None and seg.radius is not None:
                segments.append(
                    ArcSegment(
                        seg.center,
                        seg.radius,
                        seg.angle_start or 0.0,
                        seg.angle_end or 0.0,
                    )
                )
        return ModularTrackCurve(segments, closed=False)

    if gear.gear_type == "rsdl" and gear.rsdl_expression:
        spec = parse_analytic_expression(gear.rsdl_expression)
        return _curve_from_analytic_spec(spec, relation)

    if gear.gear_type == "anneau":
        return build_circle(contact_size_for_relation(gear, relation))

    return build_circle(gear.size or 1)


def _gear_perimeter(gear: GearConfig, relation: str) -> float:
    if gear.gear_type == "rsdl" and gear.rsdl_expression:
        try:
            spec = parse_analytic_expression(gear.rsdl_expression)
        except RsdlParseError:
            return float(contact_size_for_relation(gear, relation))
        curve = _curve_from_analytic_spec(spec, relation)
        return curve.length
    return float(contact_size_for_relation(gear, relation))


def generate_simple_circle_for_index(
    hole_offset: float,
    steps: int,
):
    """
    Fallback si on n’a pas assez d’engrenages : on simule un cercle
    dont le rayon dépend du trou indexé.
    On prend un rayon "référence" R_tip = 50 mm.
    Trou 0 : rayon = R_tip
    Trou n : R = R_tip - n
    R est clampé à >= 0.
    """
    R_tip = 50.0
    d = R_tip - hole_offset
    if d < 0:
        d = 0.0

    pts = []
    for i in range(steps):
        t = 2.0 * math.pi * i / (steps - 1)
        x = d * math.cos(t)
        y = d * math.sin(t)
        pts.append((x, y))
    return pts


def _rsdl_pen_local_vector(
    wheel_curve: BaseCurve,
    hole_offset: float,
    hole_direction_deg: float,
) -> Tuple[float, float]:
    if wheel_curve.length > 0:
        x0, y0, _, _ = wheel_curve.eval(0.0)
    else:
        x0, y0 = (1.0, 0.0)
    norm = math.hypot(x0, y0)
    if norm == 0:
        ux, uy = 1.0, 0.0
    else:
        ux, uy = x0 / norm, y0 / norm
    vx, vy = -uy, ux
    theta = math.radians(hole_direction_deg)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    return (
        hole_offset * (cos_t * ux + sin_t * vx),
        hole_offset * (cos_t * uy + sin_t * vy),
    )


def _rsdl_curve_center(curve: BaseCurve, samples: int = 360) -> Tuple[float, float]:
    if curve.length <= 0:
        return 0.0, 0.0
    sum_x = 0.0
    sum_y = 0.0
    for i in range(samples):
        s = curve.length * i / max(1, samples - 1)
        x, y, _, _ = curve.eval(s)
        sum_x += x
        sum_y += y
    return sum_x / samples, sum_y / samples


class _OffsetCurve(BaseCurve):
    def __init__(self, base: BaseCurve, offset: Tuple[float, float]):
        self.base = base
        self.offset = offset
        super().__init__(base.length, closed=base.closed)

    def eval(self, s: float) -> Tuple[float, float, Tuple[float, float], Tuple[float, float]]:
        x, y, tangent, normal = self.base.eval(s)
        ox, oy = self.offset
        return x - ox, y - oy, tangent, normal


class _RotateCurve(BaseCurve):
    def __init__(self, base: BaseCurve, pivot: Tuple[float, float], cos_a: float, sin_a: float):
        self.base = base
        self.pivot = pivot
        self.cos_a = cos_a
        self.sin_a = sin_a
        super().__init__(base.length, closed=base.closed)

    def eval(self, s: float) -> Tuple[float, float, Tuple[float, float], Tuple[float, float]]:
        x, y, tangent, normal = self.base.eval(s)
        px, py = self.pivot
        rel_x = x - px
        rel_y = y - py
        rot_x = rel_x * self.cos_a - rel_y * self.sin_a
        rot_y = rel_x * self.sin_a + rel_y * self.cos_a
        tx, ty = tangent
        nx, ny = normal
        rot_tx = tx * self.cos_a - ty * self.sin_a
        rot_ty = tx * self.sin_a + ty * self.cos_a
        rot_nx = nx * self.cos_a - ny * self.sin_a
        rot_ny = nx * self.sin_a + ny * self.cos_a
        return rot_x + px, rot_y + py, (rot_tx, rot_ty), (rot_nx, rot_ny)


def _align_curve_start_to_top(curve: BaseCurve) -> BaseCurve:
    if curve.length <= 0:
        return curve
    cx, cy = _rsdl_curve_center(curve)
    x0, y0, _, _ = curve.eval(0.0)
    vx = x0 - cx
    vy = y0 - cy
    if math.hypot(vx, vy) == 0:
        return curve
    current_angle = math.atan2(vy, vx)
    target_angle = math.pi / 2.0
    delta = target_angle - current_angle
    cos_a = math.cos(delta)
    sin_a = math.sin(delta)
    return _RotateCurve(curve, (cx, cy), cos_a, sin_a)


def _align_stationary_polygon_curve(gear: GearConfig, curve: Optional[BaseCurve]) -> Optional[BaseCurve]:
    if curve is None or gear.gear_type != "rsdl" or not gear.rsdl_expression:
        return curve
    try:
        spec = parse_analytic_expression(gear.rsdl_expression)
    except RsdlParseError:
        return curve
    if spec.__class__.__name__ != "PolygonSpec":
        return curve
    return _align_curve_start_to_top(curve)


def _align_base_curve_for_rsdl(
    base_curve: Optional[BaseCurve],
    stationary_gear: GearConfig,
    rolling_gear: GearConfig,
) -> Optional[BaseCurve]:
    if base_curve is None:
        return None
    if (
        rolling_gear.gear_type == "rsdl"
        and rolling_gear.rsdl_expression
        and isinstance(base_curve, CircleCurve)
    ):
        return _align_curve_start_to_top(base_curve)
    return _align_stationary_polygon_curve(stationary_gear, base_curve)


def _align_base_curve_start(
    base_curve: Optional[BaseCurve],
    stationary_gear: GearConfig,
    rolling_gear: GearConfig,
    relation: str,
) -> Optional[BaseCurve]:
    base_curve = _align_base_curve_for_rsdl(base_curve, stationary_gear, rolling_gear)
    if base_curve is None:
        return None
    if (
        relation == "dedans"
        and isinstance(base_curve, CircleCurve)
        and not (rolling_gear.gear_type == "rsdl" and rolling_gear.rsdl_expression)
    ):
        return _align_curve_start_to_top(base_curve)
    return base_curve


def generate_trochoid_points_for_layer_path(
    layer: LayerConfig,
    path: PathConfig,
    steps: int = 5000,
):
    if not layer.gears:
        return []

    g0 = layer.gears[0]
    g1 = layer.gears[1] if len(layer.gears) > 1 else None
    relation = g1.relation if g1 is not None else "stationnaire"

    r = 1.0
    wheel_size = 1.0
    base_curve = None
    wheel_curve = None
    side = 1
    pen_local = None

    if g1 is None:
        return generate_simple_circle_for_index(path.hole_offset, steps)

    if g0.gear_type == "modulaire" and g0.modular_notation:
        track = modular_tracks.build_track_from_notation(
            g0.modular_notation,
            inner_size=g0.size or 1,
            outer_size=g0.outer_size or (g0.size + 1),
            steps_per_unit=3,
        )
        segments = []
        for seg in track.segments:
            if seg.kind == "line":
                segments.append(LineSegment(seg.start, seg.end))
            elif seg.kind == "arc" and seg.center is not None and seg.radius is not None:
                segments.append(
                    ArcSegment(
                        seg.center,
                        seg.radius,
                        seg.angle_start or 0.0,
                        seg.angle_end or 0.0,
                    )
                )
        base_curve = ModularTrackCurve(segments, closed=False)
    else:
        base_curve = _curve_from_gear(g0, relation)

    if base_curve is None or base_curve.length <= 0:
        return []

    base_curve = _align_base_curve_start(base_curve, g0, g1, relation)

    if relation == "dedans":
        side = -1
    elif relation == "dehors":
        side = 1

    wheel_size = max(1.0, _gear_perimeter(g1, relation))
    r = contact_radius_for_relation(g1, relation)

    use_rsdl_wheel = g1.gear_type == "rsdl" and g1.rsdl_expression
    if use_rsdl_wheel:
        wheel_curve = _curve_from_gear(g1, relation)
    elif relation == "stationnaire":
        wheel_curve = CircleCurve(wheel_size)
    elif relation == "dedans":
        wheel_curve = CircleCurve(wheel_size)
    elif relation == "dehors":
        wheel_curve = CircleCurve(wheel_size)

    if wheel_curve is None:
        return []

    hole_offset = path.hole_offset
    hole_direction = path.hole_direction
    alpha0 = 2.0 * math.pi * (path.phase_offset or 0.0)
    if use_rsdl_wheel:
        center_offset = _rsdl_curve_center(wheel_curve)
        wheel_curve = _OffsetCurve(wheel_curve, center_offset)
        if g1.rsdl_expression:
            pen_local = _rsdl_pen_local_vector(wheel_curve, hole_offset, hole_direction)
    else:
        hole_offset = hole_offset

    phase_turns = phase_offset_turns(path.phase_offset or 0.0, int(wheel_size))
    phase_radians = 2.0 * math.pi * phase_turns
    alpha0 = phase_radians

    points = []
    for i in range(steps):
        s = base_curve.length * i / max(1, steps - 1)
        if use_rsdl_wheel:
            px, py = roll_pen_position(
                s,
                base_curve,
                wheel_curve,
                hole_offset,
                side,
                alpha0,
                epsilon=1,
                pen_local=pen_local,
            )
        else:
            px, py = pen_position(
                s,
                base_curve,
                r,
                hole_offset,
                side,
                alpha0,
                epsilon=1,
            )
        points.append((px, py))

    return points


__all__ = [
    "GearConfig",
    "LayerConfig",
    "PathConfig",
    "contact_radius_for_relation",
    "contact_size_for_relation",
    "generate_simple_circle_for_index",
    "generate_trochoid_points_for_layer_path",
    "phase_offset_turns",
    "radius_from_size",
    "_curve_from_analytic_spec",
    "_curve_from_gear",
    "_gear_perimeter",
    "_rsdl_curve_center",
    "_rsdl_pen_local_vector",
]
