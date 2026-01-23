from __future__ import annotations

import math
from typing import TYPE_CHECKING

import spirosim_math as sm

if TYPE_CHECKING:
    from spirosim_math import LayerConfig, PathConfig


def generate_trochoid_points(
    layer: "LayerConfig",
    path: "PathConfig",
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
        base_points = sm.generate_simple_circle_for_index(hole_offset, steps)
        phase_turns = sm.phase_offset_turns(path.phase_offset, 1)
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
        base_curve = sm._curve_from_gear(
            g0,
            relation,
            rsdl_curve_builder=rsdl_curve_builder,
            modular_curve_builder=modular_curve_builder,
        )
        base_curve = sm._align_base_curve_start(
            base_curve,
            g0,
            g1,
            relation,
            rsdl_is_polygon=rsdl_is_polygon,
        )
    except Exception:
        base_curve = None

    if base_curve is None or base_curve.length <= 0:
        base_points = sm.generate_simple_circle_for_index(hole_offset, steps)
        phase_turns = sm.phase_offset_turns(path.phase_offset, 1)
        total_angle = -(2.0 * math.pi * phase_turns)

        cos_a = math.cos(total_angle)
        sin_a = math.sin(total_angle)
        rotated = []
        for (x, y) in base_points:
            xr = x * cos_a - y * sin_a
            yr = x * sin_a + y * cos_a
            rotated.append((xr, yr))
        return rotated

    wheel_size = max(1.0, sm._gear_perimeter(g1, relation, rsdl_curve_builder=rsdl_curve_builder))
    r = sm.radius_from_size(wheel_size)
    if g1.gear_type == "anneau":
        tip_size = g1.outer_size or g1.size
    elif g1.gear_type == "rsdl" and g1.rsdl_expression:
        tip_size = wheel_size
    else:
        tip_size = g1.size

    if g1.gear_type == "rsdl" and g1.rsdl_expression:
        wheel_curve = None
        try:
            wheel_curve = sm._curve_from_gear(g1, relation, rsdl_curve_builder=rsdl_curve_builder)
        except Exception:
            wheel_curve = None
        tip_radius = sm._max_curve_radius(wheel_curve) if wheel_curve else sm.radius_from_size(tip_size)
    else:
        tip_radius = sm.radius_from_size(tip_size)

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
        isinstance(base_curve, sm.CircleCurve) or (g0.gear_type == "rsdl" and g0.rsdl_expression)
    ):
        alpha0 = math.pi
    else:
        alpha0 = 0.0

    base_points = []
    use_rsdl_wheel = g1.gear_type == "rsdl" and g1.rsdl_expression
    if use_rsdl_wheel:
        try:
            wheel_curve = sm._curve_from_gear(g1, relation, rsdl_curve_builder=rsdl_curve_builder)
        except Exception:
            wheel_curve = None
    else:
        wheel_curve = None
    if wheel_curve is not None and g1.gear_type == "rsdl" and g1.rsdl_expression:
        center_offset = sm._rsdl_curve_center(wheel_curve)
        wheel_curve = sm._OffsetCurve(wheel_curve, center_offset)
    if wheel_curve is not None:
        if g1.gear_type == "rsdl" and g1.rsdl_expression:
            pen_local = sm._rsdl_pen_local_vector(wheel_curve, hole_offset, hole_direction)
        else:
            pen_local = sm.wheel_pen_local_vector(base_curve, wheel_curve, d, side, alpha0)
        for i in range(steps):
            s = s_max * i / (steps - 1)
            x, y = sm.roll_pen_position(
                s, base_curve, wheel_curve, d, side, alpha0, epsilon, pen_local=pen_local
            )
            base_points.append((x, y))
    else:
        for i in range(steps):
            s = s_max * i / (steps - 1)
            x, y = sm.pen_position(s, base_curve, r, d, side, alpha0, epsilon)
            base_points.append((x, y))

    phase_turns = sm.phase_offset_turns(path.phase_offset, max(1, int(round(base_curve.length))))
    total_angle = -(2.0 * math.pi * phase_turns)

    cos_a = math.cos(total_angle)
    sin_a = math.sin(total_angle)
    rotated_points = []
    for (x, y) in base_points:
        xr = x * cos_a - y * sin_a
        yr = x * sin_a + y * cos_a
        rotated_points.append((xr, yr))

    return rotated_points
