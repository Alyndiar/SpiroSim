from __future__ import annotations

import importlib.util
import math
from typing import TYPE_CHECKING

import spirosim_math as sm
from math_backends import python_backend

if TYPE_CHECKING:
    from spirosim_math import LayerConfig, PathConfig

NUMBA_AVAILABLE = importlib.util.find_spec("numba") is not None
if NUMBA_AVAILABLE:
    import numba
    import numpy as np

    _CURVE_CIRCLE = 0
    _CURVE_LINE = 1
    _CURVE_PIECEWISE = 2


def _numba_curve_data(base_curve: sm.BaseCurve):
    if not NUMBA_AVAILABLE:
        return None
    if isinstance(base_curve, sm.CircleBaseCurve):
        base_curve = base_curve._circle
    if isinstance(base_curve, sm.CircleCurve):
        return (_CURVE_CIRCLE, float(base_curve.length), bool(base_curve.closed), ())
    if isinstance(base_curve, sm.StraightLineCurve):
        return (_CURVE_LINE, float(base_curve.length), bool(base_curve.closed), ())
    if isinstance(base_curve, sm.PiecewiseCurve):
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
            if isinstance(seg, sm.LineSegment):
                seg_kind[idx] = 0
                start_x[idx], start_y[idx] = seg.start
                end_x[idx], end_y[idx] = seg.end
            elif isinstance(seg, sm.ArcSegment):
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


if NUMBA_AVAILABLE:

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


def generate_trochoid_points(
    layer: "LayerConfig",
    path: "PathConfig",
    steps: int = 5000,
    *,
    rsdl_curve_builder=None,
    modular_curve_builder=None,
    rsdl_is_polygon=None,
):
    if not NUMBA_AVAILABLE:
        return python_backend.generate_trochoid_points(
            layer,
            path,
            steps,
            rsdl_curve_builder=rsdl_curve_builder,
            modular_curve_builder=modular_curve_builder,
            rsdl_is_polygon=rsdl_is_polygon,
        )

    hole_offset = float(path.hole_offset)

    if len(layer.gears) < 2:
        return python_backend.generate_trochoid_points(
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
        return python_backend.generate_trochoid_points(
            layer,
            path,
            steps,
            rsdl_curve_builder=rsdl_curve_builder,
            modular_curve_builder=modular_curve_builder,
            rsdl_is_polygon=rsdl_is_polygon,
        )

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

    use_rsdl_wheel = g1.gear_type == "rsdl" and g1.rsdl_expression
    if use_rsdl_wheel:
        return python_backend.generate_trochoid_points(
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

    phase_turns = sm.phase_offset_turns(path.phase_offset, max(1, int(round(base_curve.length))))
    total_angle = -(2.0 * math.pi * phase_turns)
    cos_a = math.cos(total_angle)
    sin_a = math.sin(total_angle)

    rotated_points = []
    for x, y in zip(px, py):
        xr = float(x) * cos_a - float(y) * sin_a
        yr = float(x) * sin_a + float(y) * cos_a
        rotated_points.append((xr, yr))

    return rotated_points
