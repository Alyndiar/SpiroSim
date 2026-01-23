from __future__ import annotations

from dataclasses import dataclass, field
import importlib.util
import math
from typing import Callable, List, Optional, Tuple

from shape_geometry import (
    ArcSegment,
    BaseCurve,
    CircleCurve,
    CircleBaseCurve,
    EllipseCurve,
    LineSegment,
    ModularTrackCurve,
    PiecewiseCurve,
    StraightLineCurve,
    build_circle,
    build_drop,
    build_oblong,
    build_rounded_polygon,
    pen_position,
    roll_pen_position,
    wheel_orientation,
    wheel_pen_local_vector,
)

Point = Tuple[float, float]
_NUMBA_AVAILABLE = importlib.util.find_spec("numba") is not None
if _NUMBA_AVAILABLE:
    import numpy as np
    import numba

    _CURVE_CIRCLE = 0
    _CURVE_LINE = 1
    _CURVE_PIECEWISE = 2


@dataclass(frozen=True)
class MathBackend:
    name: str
    label: str
    available: bool
    generator: Callable


_BACKENDS: dict[str, MathBackend] = {}
_ACTIVE_BACKEND = "python"


def register_backend(backend: MathBackend) -> None:
    _BACKENDS[backend.name] = backend


def list_backends(*, available_only: bool = False) -> list[MathBackend]:
    backends = list(_BACKENDS.values())
    if available_only:
        backends = [b for b in backends if b.available]
    return sorted(backends, key=lambda b: b.name)


def get_backend_name() -> str:
    return _ACTIVE_BACKEND


def set_backend(name: str) -> None:
    backend = _BACKENDS.get(name)
    if backend is None:
        raise ValueError(f"Unknown math backend: {name}")
    if not backend.available:
        raise ValueError(f"Math backend not available: {name}")
    global _ACTIVE_BACKEND
    _ACTIVE_BACKEND = backend.name


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


def _curve_from_gear(
    gear: GearConfig,
    relation: str,
    *,
    rsdl_curve_builder=None,
    modular_curve_builder=None,
) -> Optional[BaseCurve]:
    if gear.gear_type == "modulaire" and gear.modular_notation and modular_curve_builder:
        return modular_curve_builder(
            gear.modular_notation,
            inner_size=gear.size or 1,
            outer_size=gear.outer_size or (gear.size + 1),
        )

    if gear.gear_type == "rsdl" and gear.rsdl_expression and rsdl_curve_builder:
        return rsdl_curve_builder(gear.rsdl_expression, relation)

    if gear.gear_type == "anneau":
        return build_circle(contact_size_for_relation(gear, relation))

    return build_circle(gear.size or 1)


def _gear_perimeter(
    gear: GearConfig,
    relation: str,
    *,
    rsdl_curve_builder=None,
) -> float:
    if gear.gear_type == "rsdl" and gear.rsdl_expression and rsdl_curve_builder:
        curve = rsdl_curve_builder(gear.rsdl_expression, relation)
        if curve is not None:
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


def _max_curve_radius(curve: Optional[BaseCurve], samples: int = 720) -> float:
    if curve is None or curve.length <= 0:
        return 0.0
    max_radius = 0.0
    for i in range(samples):
        s = curve.length * i / max(1, samples - 1)
        x, y, _, _ = curve.eval(s)
        max_radius = max(max_radius, math.hypot(x, y))
    return max_radius


def _numba_curve_data(base_curve: BaseCurve):
    if not _NUMBA_AVAILABLE:
        return None
    if isinstance(base_curve, CircleBaseCurve):
        base_curve = base_curve._circle
    if isinstance(base_curve, CircleCurve):
        return (_CURVE_CIRCLE, float(base_curve.length), bool(base_curve.closed), ())
    if isinstance(base_curve, StraightLineCurve):
        return (_CURVE_LINE, float(base_curve.length), bool(base_curve.closed), ())
    if isinstance(base_curve, PiecewiseCurve):
        seg_count = len(base_curve.segments)
        if seg_count == 0:
            return None
        seg_ends = np.array(base_curve._segment_ends, dtype=np.float64)
        seg_kind = np.zeros(seg_count, dtype=np.int64)
        start_x = np.zeros(seg_count, dtype=np.float64)
        start_y = np.zeros(seg_count, dtype=np.float64)
        end_x = np.zeros(seg_count, dtype=np.float64)
        end_y = np.zeros(seg_count, dtype=np.float64)
        center_x = np.zeros(seg_count, dtype=np.float64)
        center_y = np.zeros(seg_count, dtype=np.float64)
        radius = np.zeros(seg_count, dtype=np.float64)
        angle_start = np.zeros(seg_count, dtype=np.float64)
        angle_end = np.zeros(seg_count, dtype=np.float64)
        for idx, seg in enumerate(base_curve.segments):
            if isinstance(seg, LineSegment):
                seg_kind[idx] = 0
                start_x[idx], start_y[idx] = seg.start
                end_x[idx], end_y[idx] = seg.end
            elif isinstance(seg, ArcSegment):
                seg_kind[idx] = 1
                center_x[idx], center_y[idx] = seg.center
                radius[idx] = float(seg.radius)
                angle_start[idx] = float(seg.angle_start)
                angle_end[idx] = float(seg.angle_end)
            else:
                return None
        return (
            _CURVE_PIECEWISE,
            float(base_curve.length),
            bool(base_curve.closed),
            (
                seg_ends,
                seg_kind,
                start_x,
                start_y,
                end_x,
                end_y,
                center_x,
                center_y,
                radius,
                angle_start,
                angle_end,
            ),
        )
    return None


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


def _align_stationary_polygon_curve(
    gear: GearConfig,
    curve: Optional[BaseCurve],
    rsdl_is_polygon=None,
) -> Optional[BaseCurve]:
    if curve is None or gear.gear_type != "rsdl" or not gear.rsdl_expression:
        return curve
    if rsdl_is_polygon and rsdl_is_polygon(gear.rsdl_expression):
        return _align_curve_start_to_top(curve)
    return curve


def _align_base_curve_for_rsdl(
    base_curve: Optional[BaseCurve],
    stationary_gear: GearConfig,
    rolling_gear: GearConfig,
    *,
    rsdl_is_polygon=None,
) -> Optional[BaseCurve]:
    if base_curve is None:
        return None
    if (
        rolling_gear.gear_type == "rsdl"
        and rolling_gear.rsdl_expression
        and isinstance(base_curve, CircleCurve)
    ):
        return _align_curve_start_to_top(base_curve)
    return _align_stationary_polygon_curve(
        stationary_gear,
        base_curve,
        rsdl_is_polygon=rsdl_is_polygon,
    )


def _align_base_curve_start(
    base_curve: Optional[BaseCurve],
    stationary_gear: GearConfig,
    rolling_gear: GearConfig,
    relation: str,
    *,
    rsdl_is_polygon=None,
) -> Optional[BaseCurve]:
    base_curve = _align_base_curve_for_rsdl(
        base_curve,
        stationary_gear,
        rolling_gear,
        rsdl_is_polygon=rsdl_is_polygon,
    )
    if base_curve is None:
        return None
    if (
        relation == "dedans"
        and isinstance(base_curve, CircleCurve)
        and not (rolling_gear.gear_type == "rsdl" and rolling_gear.rsdl_expression)
    ):
        return _align_curve_start_to_top(base_curve)
    return base_curve


def _generate_trochoid_points_python(
    layer: LayerConfig,
    path: PathConfig,
    steps: int = 5000,
    *,
    rsdl_curve_builder=None,
    modular_curve_builder=None,
    rsdl_is_polygon=None,
):
    """
    Génère la courbe pour un path donné, en utilisant la configuration
    du layer (engrenages + organisation).

    Convention :
      - Le PREMIER engrenage de la couche (gears[0]) est stationnaire
        et centré en (0, 0).
      - Le DEUXIÈME engrenage (gears[1]) est mobile et porte les trous du path.
      - path.hole_offset est un float, peut être négatif.
    """
    hole_offset = float(path.hole_offset)
    hole_direction = float(getattr(path, "hole_direction", 0.0))

    if len(layer.gears) < 2:
        base_points = generate_simple_circle_for_index(hole_offset, steps)
        phase_turns = phase_offset_turns(path.phase_offset, 1)
        total_angle = -(2.0 * math.pi * phase_turns)

        cos_a = math.cos(total_angle)
        sin_a = math.sin(total_angle)
        rotated = []
        for (x, y) in base_points:
            xr = x * cos_a - y * sin_a
            yr = x * sin_a + y * cos_a
            rotated.append((xr, yr))
        return rotated

    g0 = layer.gears[0]
    g1 = layer.gears[1]
    relation = g1.relation

    try:
        base_curve = _curve_from_gear(
            g0,
            relation,
            rsdl_curve_builder=rsdl_curve_builder,
            modular_curve_builder=modular_curve_builder,
        )
        base_curve = _align_base_curve_start(
            base_curve,
            g0,
            g1,
            relation,
            rsdl_is_polygon=rsdl_is_polygon,
        )
    except Exception:
        base_curve = None

    if base_curve is None or base_curve.length <= 0:
        base_points = generate_simple_circle_for_index(hole_offset, steps)
        phase_turns = phase_offset_turns(path.phase_offset, 1)
        total_angle = -(2.0 * math.pi * phase_turns)

        cos_a = math.cos(total_angle)
        sin_a = math.sin(total_angle)
        rotated = []
        for (x, y) in base_points:
            xr = x * cos_a - y * sin_a
            yr = x * sin_a + y * cos_a
            rotated.append((xr, yr))
        return rotated

    wheel_size = max(1.0, _gear_perimeter(g1, relation, rsdl_curve_builder=rsdl_curve_builder))
    r = radius_from_size(wheel_size)
    if g1.gear_type == "anneau":
        tip_size = g1.outer_size or g1.size
    elif g1.gear_type == "rsdl" and g1.rsdl_expression:
        tip_size = wheel_size
    else:
        tip_size = g1.size

    if g1.gear_type == "rsdl" and g1.rsdl_expression:
        wheel_curve = None
        try:
            wheel_curve = _curve_from_gear(g1, relation, rsdl_curve_builder=rsdl_curve_builder)
        except Exception:
            wheel_curve = None
        tip_radius = _max_curve_radius(wheel_curve) if wheel_curve else radius_from_size(tip_size)
    else:
        tip_radius = radius_from_size(tip_size)

    if g1.gear_type == "rsdl" and g1.rsdl_expression:
        d = hole_offset
    else:
        d = tip_radius - hole_offset

    if base_curve.closed:
        g = math.gcd(int(round(base_curve.length)), int(round(wheel_size))) or 1
        s_max = base_curve.length * (wheel_size / g)
    else:
        s_max = base_curve.length
    s_max = max(s_max, base_curve.length)

    side = 1 if relation == "dedans" else -1
    epsilon = side
    if relation == "dedans" and (
        isinstance(base_curve, CircleCurve) or (g0.gear_type == "rsdl" and g0.rsdl_expression)
    ):
        alpha0 = math.pi
    else:
        alpha0 = 0.0

    base_points = []
    use_rsdl_wheel = g1.gear_type == "rsdl" and g1.rsdl_expression
    if use_rsdl_wheel:
        try:
            wheel_curve = _curve_from_gear(g1, relation, rsdl_curve_builder=rsdl_curve_builder)
        except Exception:
            wheel_curve = None
    else:
        wheel_curve = None
    if wheel_curve is not None and g1.gear_type == "rsdl" and g1.rsdl_expression:
        center_offset = _rsdl_curve_center(wheel_curve)
        wheel_curve = _OffsetCurve(wheel_curve, center_offset)
    if wheel_curve is not None:
        if g1.gear_type == "rsdl" and g1.rsdl_expression:
            pen_local = _rsdl_pen_local_vector(wheel_curve, hole_offset, hole_direction)
        else:
            pen_local = wheel_pen_local_vector(base_curve, wheel_curve, d, side, alpha0)
        for i in range(steps):
            s = s_max * i / (steps - 1)
            x, y = roll_pen_position(s, base_curve, wheel_curve, d, side, alpha0, epsilon, pen_local=pen_local)
            base_points.append((x, y))
    else:
        for i in range(steps):
            s = s_max * i / (steps - 1)
            x, y = pen_position(s, base_curve, r, d, side, alpha0, epsilon)
            base_points.append((x, y))

    phase_turns = phase_offset_turns(path.phase_offset, max(1, int(round(base_curve.length))))
    total_angle = -(2.0 * math.pi * phase_turns)

    cos_a = math.cos(total_angle)
    sin_a = math.sin(total_angle)
    rotated_points = []
    for (x, y) in base_points:
        xr = x * cos_a - y * sin_a
        yr = x * sin_a + y * cos_a
        rotated_points.append((xr, yr))

    return rotated_points


def _generate_trochoid_points_numba(
    layer: LayerConfig,
    path: PathConfig,
    steps: int = 5000,
    *,
    rsdl_curve_builder=None,
    modular_curve_builder=None,
    rsdl_is_polygon=None,
):
    if not _NUMBA_AVAILABLE:
        return _generate_trochoid_points_python(
            layer,
            path,
            steps,
            rsdl_curve_builder=rsdl_curve_builder,
            modular_curve_builder=modular_curve_builder,
            rsdl_is_polygon=rsdl_is_polygon,
        )

    hole_offset = float(path.hole_offset)

    if len(layer.gears) < 2:
        return _generate_trochoid_points_python(
            layer,
            path,
            steps,
            rsdl_curve_builder=rsdl_curve_builder,
            modular_curve_builder=modular_curve_builder,
            rsdl_is_polygon=rsdl_is_polygon,
        )

    g0 = layer.gears[0]
    g1 = layer.gears[1]
    relation = g1.relation

    try:
        base_curve = _curve_from_gear(
            g0,
            relation,
            rsdl_curve_builder=rsdl_curve_builder,
            modular_curve_builder=modular_curve_builder,
        )
        base_curve = _align_base_curve_start(
            base_curve,
            g0,
            g1,
            relation,
            rsdl_is_polygon=rsdl_is_polygon,
        )
    except Exception:
        base_curve = None

    if base_curve is None or base_curve.length <= 0:
        return _generate_trochoid_points_python(
            layer,
            path,
            steps,
            rsdl_curve_builder=rsdl_curve_builder,
            modular_curve_builder=modular_curve_builder,
            rsdl_is_polygon=rsdl_is_polygon,
        )

    wheel_size = max(1.0, _gear_perimeter(g1, relation, rsdl_curve_builder=rsdl_curve_builder))
    r = radius_from_size(wheel_size)
    if g1.gear_type == "anneau":
        tip_size = g1.outer_size or g1.size
    elif g1.gear_type == "rsdl" and g1.rsdl_expression:
        tip_size = wheel_size
    else:
        tip_size = g1.size

    if g1.gear_type == "rsdl" and g1.rsdl_expression:
        wheel_curve = None
        try:
            wheel_curve = _curve_from_gear(g1, relation, rsdl_curve_builder=rsdl_curve_builder)
        except Exception:
            wheel_curve = None
        tip_radius = _max_curve_radius(wheel_curve) if wheel_curve else radius_from_size(tip_size)
    else:
        tip_radius = radius_from_size(tip_size)

    if g1.gear_type == "rsdl" and g1.rsdl_expression:
        d = hole_offset
    else:
        d = tip_radius - hole_offset

    if base_curve.closed:
        g = math.gcd(int(round(base_curve.length)), int(round(wheel_size))) or 1
        s_max = base_curve.length * (wheel_size / g)
    else:
        s_max = base_curve.length
    s_max = max(s_max, base_curve.length)

    side = 1 if relation == "dedans" else -1
    epsilon = side
    if relation == "dedans" and (
        isinstance(base_curve, CircleCurve) or (g0.gear_type == "rsdl" and g0.rsdl_expression)
    ):
        alpha0 = math.pi
    else:
        alpha0 = 0.0

    use_rsdl_wheel = g1.gear_type == "rsdl" and g1.rsdl_expression
    if use_rsdl_wheel:
        return _generate_trochoid_points_python(
            layer,
            path,
            steps,
            rsdl_curve_builder=rsdl_curve_builder,
            modular_curve_builder=modular_curve_builder,
            rsdl_is_polygon=rsdl_is_polygon,
        )

    s_values = np.linspace(0.0, s_max, steps, dtype=np.float64)
    curve_data = _numba_curve_data(base_curve)
    if curve_data is None:
        xb = np.empty(steps, dtype=np.float64)
        yb = np.empty(steps, dtype=np.float64)
        tx = np.empty(steps, dtype=np.float64)
        ty = np.empty(steps, dtype=np.float64)
        nx = np.empty(steps, dtype=np.float64)
        ny = np.empty(steps, dtype=np.float64)
        for i, s in enumerate(s_values):
            x, y, tangent, normal = base_curve.eval(float(s))
            xb[i] = x
            yb[i] = y
            tx[i] = tangent[0]
            ty[i] = tangent[1]
            nx[i] = normal[0]
            ny[i] = normal[1]
    else:
        curve_kind, curve_length, curve_closed, curve_arrays = curve_data
        if curve_kind == _CURVE_PIECEWISE:
            (
                seg_ends,
                seg_kind,
                start_x,
                start_y,
                end_x,
                end_y,
                center_x,
                center_y,
                radius,
                angle_start,
                angle_end,
            ) = curve_arrays
        else:
            seg_ends = np.empty(0, dtype=np.float64)
            seg_kind = np.empty(0, dtype=np.int64)
            start_x = np.empty(0, dtype=np.float64)
            start_y = np.empty(0, dtype=np.float64)
            end_x = np.empty(0, dtype=np.float64)
            end_y = np.empty(0, dtype=np.float64)
            center_x = np.empty(0, dtype=np.float64)
            center_y = np.empty(0, dtype=np.float64)
            radius = np.empty(0, dtype=np.float64)
            angle_start = np.empty(0, dtype=np.float64)
            angle_end = np.empty(0, dtype=np.float64)
        xb, yb, tx, ty, nx, ny = _eval_curve_points_numba(
            s_values,
            curve_kind,
            curve_length,
            curve_closed,
            seg_ends,
            seg_kind,
            start_x,
            start_y,
            end_x,
            end_y,
            center_x,
            center_y,
            radius,
            angle_start,
            angle_end,
        )

    px, py = _pen_positions_numba(s_values, xb, yb, tx, ty, nx, ny, r, d, side, alpha0, epsilon)

    phase_turns = phase_offset_turns(path.phase_offset, max(1, int(round(base_curve.length))))
    total_angle = -(2.0 * math.pi * phase_turns)
    cos_a = math.cos(total_angle)
    sin_a = math.sin(total_angle)

    rotated_points = []
    for x, y in zip(px, py):
        xr = float(x) * cos_a - float(y) * sin_a
        yr = float(x) * sin_a + float(y) * cos_a
        rotated_points.append((xr, yr))

    return rotated_points


if _NUMBA_AVAILABLE:

    @numba.njit(cache=True)
    def _eval_curve_points_numba(
        s_values: np.ndarray,
        curve_kind: int,
        curve_length: float,
        curve_closed: bool,
        seg_ends: np.ndarray,
        seg_kind: np.ndarray,
        start_x: np.ndarray,
        start_y: np.ndarray,
        end_x: np.ndarray,
        end_y: np.ndarray,
        center_x: np.ndarray,
        center_y: np.ndarray,
        radius: np.ndarray,
        angle_start: np.ndarray,
        angle_end: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n = len(s_values)
        xb = np.empty(n, dtype=np.float64)
        yb = np.empty(n, dtype=np.float64)
        tx = np.empty(n, dtype=np.float64)
        ty = np.empty(n, dtype=np.float64)
        nx = np.empty(n, dtype=np.float64)
        ny = np.empty(n, dtype=np.float64)
        for i in range(n):
            s = s_values[i]
            if curve_length <= 0.0:
                x = 0.0
                y = 0.0
                t_x = 1.0
                t_y = 0.0
            else:
                if curve_closed:
                    s = s % curve_length
                elif s < 0.0:
                    s = 0.0
                elif s > curve_length:
                    s = curve_length

                if curve_kind == _CURVE_CIRCLE:
                    radius_local = curve_length / (2.0 * math.pi)
                    if radius_local == 0.0:
                        x = 0.0
                        y = 0.0
                        t_x = 1.0
                        t_y = 0.0
                    else:
                        theta = s / radius_local
                        x = radius_local * math.cos(theta)
                        y = radius_local * math.sin(theta)
                        t_x = -math.sin(theta)
                        t_y = math.cos(theta)
                elif curve_kind == _CURVE_LINE:
                    x = s
                    y = 0.0
                    t_x = 1.0
                    t_y = 0.0
                else:
                    idx = 0
                    for j in range(len(seg_ends)):
                        if s <= seg_ends[j]:
                            idx = j
                            break
                        idx = j
                    seg_start = 0.0 if idx == 0 else seg_ends[idx - 1]
                    local_s = s - seg_start
                    if seg_kind[idx] == 0:
                        dx = end_x[idx] - start_x[idx]
                        dy = end_y[idx] - start_y[idx]
                        seg_len = math.hypot(dx, dy)
                        if seg_len == 0.0:
                            t = 0.0
                        else:
                            t = local_s / seg_len
                        x = start_x[idx] + dx * t
                        y = start_y[idx] + dy * t
                        if seg_len == 0.0:
                            t_x = 1.0
                            t_y = 0.0
                        else:
                            t_x = dx / seg_len
                            t_y = dy / seg_len
                    else:
                        delta = angle_end[idx] - angle_start[idx]
                        seg_len = abs(radius[idx] * delta)
                        if seg_len == 0.0:
                            t = 0.0
                        else:
                            t = local_s / seg_len
                        angle = angle_start[idx] + delta * t
                        x = center_x[idx] + radius[idx] * math.cos(angle)
                        y = center_y[idx] + radius[idx] * math.sin(angle)
                        sign = 1.0 if delta >= 0.0 else -1.0
                        t_x = -math.sin(angle) * sign
                        t_y = math.cos(angle) * sign

            n_x = -t_y
            n_y = t_x
            xb[i] = x
            yb[i] = y
            tx[i] = t_x
            ty[i] = t_y
            nx[i] = n_x
            ny[i] = n_y

        return xb, yb, tx, ty, nx, ny


    @numba.njit(cache=True)
    def _pen_positions_numba(
        s_values: np.ndarray,
        xb: np.ndarray,
        yb: np.ndarray,
        tx: np.ndarray,
        ty: np.ndarray,
        nx: np.ndarray,
        ny: np.ndarray,
        r: float,
        d: float,
        side: int,
        alpha0: float,
        epsilon: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        n = len(s_values)
        out_x = np.empty(n, dtype=np.float64)
        out_y = np.empty(n, dtype=np.float64)
        for i in range(n):
            alpha = alpha0 + epsilon * s_values[i] / r
            cos_a = math.cos(alpha)
            sin_a = math.sin(alpha)
            cx = xb[i] + side * r * nx[i]
            cy = yb[i] + side * r * ny[i]
            out_x[i] = cx + d * (cos_a * nx[i] + sin_a * tx[i])
            out_y[i] = cy + d * (cos_a * ny[i] + sin_a * ty[i])
        return out_x, out_y


def generate_trochoid_points_for_layer_path(
    layer: LayerConfig,
    path: PathConfig,
    steps: int = 5000,
    *,
    rsdl_curve_builder=None,
    modular_curve_builder=None,
    rsdl_is_polygon=None,
):
    backend = _BACKENDS.get(_ACTIVE_BACKEND)
    if backend is None:
        raise ValueError(f"Unknown math backend: {_ACTIVE_BACKEND}")
    return backend.generator(
        layer,
        path,
        steps,
        rsdl_curve_builder=rsdl_curve_builder,
        modular_curve_builder=modular_curve_builder,
        rsdl_is_polygon=rsdl_is_polygon,
    )


register_backend(
    MathBackend(
        name="python",
        label="Python",
        available=True,
        generator=_generate_trochoid_points_python,
    )
)
register_backend(
    MathBackend(
        name="numba",
        label="Numba",
        available=_NUMBA_AVAILABLE,
        generator=_generate_trochoid_points_numba,
    )
)


__all__ = [
    "ArcSegment",
    "BaseCurve",
    "CircleCurve",
    "GearConfig",
    "LayerConfig",
    "LineSegment",
    "MathBackend",
    "ModularTrackCurve",
    "PathConfig",
    "contact_radius_for_relation",
    "contact_size_for_relation",
    "generate_simple_circle_for_index",
    "generate_trochoid_points_for_layer_path",
    "get_backend_name",
    "list_backends",
    "pen_position",
    "phase_offset_turns",
    "radius_from_size",
    "register_backend",
    "roll_pen_position",
    "set_backend",
    "wheel_orientation",
    "wheel_pen_local_vector",
    "_curve_from_analytic_spec",
    "_curve_from_gear",
    "_gear_perimeter",
    "_rsdl_curve_center",
    "_rsdl_pen_local_vector",
]
