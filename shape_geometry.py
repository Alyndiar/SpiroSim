from __future__ import annotations

from dataclasses import dataclass
import math
from bisect import bisect_left
from typing import Iterable, List, Tuple

Point = Tuple[float, float]


def _normalize(x: float, y: float) -> Tuple[float, float]:
    n = math.hypot(x, y)
    if n == 0:
        return (0.0, 0.0)
    return (x / n, y / n)


def _rotate(x: float, y: float, cos_a: float, sin_a: float) -> Tuple[float, float]:
    return (x * cos_a - y * sin_a, x * sin_a + y * cos_a)


def _rotation_from_to(
    tx: float,
    ty: float,
    ux: float,
    uy: float,
) -> Tuple[float, float]:
    tx, ty = _normalize(tx, ty)
    ux, uy = _normalize(ux, uy)
    cos_a = tx * ux + ty * uy
    sin_a = tx * uy - ty * ux
    return cos_a, sin_a


class BaseCurve:
    def __init__(self, length: float, closed: bool = True) -> None:
        self.length = max(0.0, length)
        self.closed = closed

    def eval(self, s: float) -> Tuple[float, float, Tuple[float, float], Tuple[float, float]]:
        raise NotImplementedError


@dataclass
class LineSegment:
    start: Point
    end: Point

    @property
    def length(self) -> float:
        return math.hypot(self.end[0] - self.start[0], self.end[1] - self.start[1])

    def eval(self, s: float) -> Tuple[Point, Tuple[float, float]]:
        length = self.length
        t = 0.0 if length == 0 else s / length
        x = self.start[0] + (self.end[0] - self.start[0]) * t
        y = self.start[1] + (self.end[1] - self.start[1]) * t
        tx, ty = _normalize(self.end[0] - self.start[0], self.end[1] - self.start[1])
        return (x, y), (tx, ty)


@dataclass
class ArcSegment:
    center: Point
    radius: float
    angle_start: float
    angle_end: float

    @property
    def length(self) -> float:
        return abs(self.radius * (self.angle_end - self.angle_start))

    def eval(self, s: float) -> Tuple[Point, Tuple[float, float]]:
        delta = self.angle_end - self.angle_start
        length = self.length
        t = 0.0 if length == 0 else s / length
        angle = self.angle_start + delta * t
        x = self.center[0] + self.radius * math.cos(angle)
        y = self.center[1] + self.radius * math.sin(angle)
        sign = 1.0 if delta >= 0 else -1.0
        tx, ty = _normalize(-math.sin(angle) * sign, math.cos(angle) * sign)
        return (x, y), (tx, ty)


class PiecewiseCurve(BaseCurve):
    def __init__(self, segments: List[LineSegment | ArcSegment], closed: bool = True) -> None:
        self.segments = segments
        self._segment_ends: List[float] = []
        total = 0.0
        for seg in segments:
            total += seg.length
            self._segment_ends.append(total)
        super().__init__(total, closed=closed)

    def _resolve_s(self, s: float) -> float:
        if self.length == 0:
            return 0.0
        if self.closed:
            return s % self.length
        return max(0.0, min(s, self.length))

    def eval(self, s: float) -> Tuple[float, float, Tuple[float, float], Tuple[float, float]]:
        if not self.segments:
            return 0.0, 0.0, (1.0, 0.0), (0.0, 1.0)
        s = self._resolve_s(s)
        idx = bisect_left(self._segment_ends, s)
        idx = min(idx, len(self.segments) - 1)
        seg_start = 0.0 if idx == 0 else self._segment_ends[idx - 1]
        local_s = s - seg_start
        point, tangent = self.segments[idx].eval(local_s)
        tx, ty = _normalize(*tangent)
        nx, ny = -ty, tx
        return point[0], point[1], (tx, ty), (nx, ny)


class CircleCurve(BaseCurve):
    def __init__(self, perimeter: float) -> None:
        self.perimeter = max(0.0, perimeter)
        self.radius = self.perimeter / (2.0 * math.pi) if self.perimeter else 0.0
        super().__init__(self.perimeter, closed=True)

    def eval(self, s: float) -> Tuple[float, float, Tuple[float, float], Tuple[float, float]]:
        if self.radius == 0:
            return 0.0, 0.0, (1.0, 0.0), (0.0, 1.0)
        s = s % self.perimeter
        theta = s / self.radius
        x = self.radius * math.cos(theta)
        y = self.radius * math.sin(theta)
        tx, ty = _normalize(-math.sin(theta), math.cos(theta))
        nx, ny = -ty, tx
        return x, y, (tx, ty), (nx, ny)


class EllipseCurve(BaseCurve):
    def __init__(self, perimeter: float, axis_a: float, axis_b: float, samples: int = 2000) -> None:
        self.perimeter = max(0.0, perimeter)
        self.axis_a = abs(axis_a)
        self.axis_b = abs(axis_b)
        self.samples = max(200, samples)
        self._t_values: List[float] = []
        self._s_values: List[float] = []
        self._scale = 1.0
        self._build_lut()
        super().__init__(self.perimeter, closed=True)

    def _build_lut(self) -> None:
        if self.axis_a == 0 or self.axis_b == 0:
            self._t_values = [0.0, 2.0 * math.pi]
            self._s_values = [0.0, 0.0]
            self._scale = 1.0
            return
        t_values = [2.0 * math.pi * i / (self.samples - 1) for i in range(self.samples)]
        points = [(self.axis_a * math.cos(t), self.axis_b * math.sin(t)) for t in t_values]
        s_values = [0.0]
        total = 0.0
        for i in range(1, len(points)):
            dx = points[i][0] - points[i - 1][0]
            dy = points[i][1] - points[i - 1][1]
            total += math.hypot(dx, dy)
            s_values.append(total)
        self._scale = 1.0 if total == 0 else (self.perimeter / total)
        self._t_values = t_values
        self._s_values = [s * self._scale for s in s_values]

    def eval(self, s: float) -> Tuple[float, float, Tuple[float, float], Tuple[float, float]]:
        if self.perimeter == 0:
            return 0.0, 0.0, (1.0, 0.0), (0.0, 1.0)
        s = s % self.perimeter
        idx = bisect_left(self._s_values, s)
        idx = max(1, min(idx, len(self._s_values) - 1))
        s0 = self._s_values[idx - 1]
        s1 = self._s_values[idx]
        t0 = self._t_values[idx - 1]
        t1 = self._t_values[idx]
        t = t0 if s1 == s0 else t0 + (t1 - t0) * ((s - s0) / (s1 - s0))
        x = self._scale * self.axis_a * math.cos(t)
        y = self._scale * self.axis_b * math.sin(t)
        dx_dt = -self._scale * self.axis_a * math.sin(t)
        dy_dt = self._scale * self.axis_b * math.cos(t)
        tx, ty = _normalize(dx_dt, dy_dt)
        nx, ny = -ty, tx
        return x, y, (tx, ty), (nx, ny)


class PathBuilder:
    def __init__(self) -> None:
        self._segments: List[LineSegment | ArcSegment] = []
        self._x = 0.0
        self._y = 0.0
        self._heading = 0.0

    @property
    def segments(self) -> List[LineSegment | ArcSegment]:
        return self._segments

    def line(self, length: float) -> None:
        dx = length * math.cos(self._heading)
        dy = length * math.sin(self._heading)
        start = (self._x, self._y)
        end = (self._x + dx, self._y + dy)
        self._segments.append(LineSegment(start, end))
        self._x, self._y = end

    def arc(self, radius: float, sweep_deg: float) -> None:
        delta = math.radians(sweep_deg)
        if radius == 0 or delta == 0:
            return
        left_x = -math.sin(self._heading)
        left_y = math.cos(self._heading)
        sign = 1.0 if delta >= 0 else -1.0
        center = (
            self._x + left_x * radius * sign,
            self._y + left_y * radius * sign,
        )
        angle_start = math.atan2(self._y - center[1], self._x - center[0])
        angle_end = angle_start + delta
        self._segments.append(ArcSegment(center, radius, angle_start, angle_end))
        self._x = center[0] + radius * math.cos(angle_end)
        self._y = center[1] + radius * math.sin(angle_end)
        self._heading += delta


def build_piecewise_from_builder(builder: PathBuilder, closed: bool = True) -> PiecewiseCurve:
    return PiecewiseCurve(builder.segments, closed=closed)


def build_rounded_polygon(perimeter: float, sides: int, side_size: float, corner_size: float) -> PiecewiseCurve:
    if sides < 2:
        return PiecewiseCurve([], closed=True)
    if side_size == corner_size:
        return build_circle(perimeter)
    r_side = side_size / (2.0 * math.pi)
    r_corner = corner_size / (2.0 * math.pi)
    if sides == 2:
        alpha = 180.0 * (perimeter - corner_size) / (side_size - corner_size)
        beta = 180.0 - alpha
        sequence = [(r_side, alpha), (r_corner, beta), (r_side, alpha), (r_corner, beta)]
    else:
        alpha = 360.0 * (perimeter - corner_size) / (sides * (side_size - corner_size))
        beta = 360.0 / sides - alpha
        sequence = []
        for _ in range(sides):
            sequence.append((r_side, alpha))
            sequence.append((r_corner, beta))
    builder = PathBuilder()
    for radius, sweep in sequence:
        builder.arc(radius, sweep)
    return build_piecewise_from_builder(builder, closed=True)


def build_drop(perimeter: float, opposite: float, half: float, link: float) -> PiecewiseCurve:
    r_half = half / (2.0 * math.pi)
    r_link = link / (2.0 * math.pi)
    r_opposite = opposite / (2.0 * math.pi)
    sequence = [(r_half, 180.0), (r_link, 60.0), (r_opposite, 60.0), (r_link, 60.0)]
    builder = PathBuilder()
    for radius, sweep in sequence:
        builder.arc(radius, sweep)
    return build_piecewise_from_builder(builder, closed=True)


def build_oblong(perimeter: float, cap_size: float) -> PiecewiseCurve:
    r_cap = cap_size / (2.0 * math.pi)
    straight = max(0.0, (perimeter - cap_size) / 2.0)
    builder = PathBuilder()
    builder.arc(r_cap, 180.0)
    builder.line(straight)
    builder.arc(r_cap, 180.0)
    builder.line(straight)
    return build_piecewise_from_builder(builder, closed=True)


def build_circle(perimeter: float) -> CircleCurve:
    return CircleCurve(perimeter)


class StraightLineCurve(BaseCurve):
    def __init__(self, length: float) -> None:
        self._length = max(0.0, length)
        super().__init__(self._length, closed=False)

    def eval(self, s: float) -> Tuple[float, float, Tuple[float, float], Tuple[float, float]]:
        s = max(0.0, min(s, self._length))
        x = s
        y = 0.0
        tx, ty = 1.0, 0.0
        nx, ny = -ty, tx
        return x, y, (tx, ty), (nx, ny)


class CircleBaseCurve(BaseCurve):
    def __init__(self, perimeter: float) -> None:
        self._circle = CircleCurve(perimeter)
        super().__init__(self._circle.length, closed=True)

    def eval(self, s: float) -> Tuple[float, float, Tuple[float, float], Tuple[float, float]]:
        return self._circle.eval(s)


def pen_position(
    s: float,
    base_curve: BaseCurve,
    r: float,
    d: float,
    side: int,
    alpha0: float,
    epsilon: int = 1,
) -> Tuple[float, float]:
    xb, yb, (tx, ty), (nx, ny) = base_curve.eval(s)
    cx = xb + side * r * nx
    cy = yb + side * r * ny
    alpha = alpha0 + epsilon * s / r
    cos_a = math.cos(alpha)
    sin_a = math.sin(alpha)
    px = cx + d * (cos_a * nx + sin_a * tx)
    py = cy + d * (cos_a * ny + sin_a * ty)
    return px, py


def wheel_orientation(
    s: float,
    base_curve: BaseCurve,
    wheel_curve: BaseCurve,
    side: int,
    epsilon: int = 1,
) -> Tuple[float, float]:
    _, _, (tx, ty), _ = base_curve.eval(s)
    target_tx, target_ty = (tx, ty) if side == 1 else (-tx, -ty)
    if wheel_curve.length > 0:
        wheel_s = epsilon * s
        if wheel_curve.closed:
            wheel_s %= wheel_curve.length
        else:
            wheel_s = max(0.0, min(wheel_s, wheel_curve.length))
    else:
        wheel_s = 0.0
    _, _, (twx, twy), _ = wheel_curve.eval(wheel_s)
    return _rotation_from_to(twx, twy, target_tx, target_ty)


def wheel_pen_local_vector(
    base_curve: BaseCurve,
    wheel_curve: BaseCurve,
    d: float,
    side: int,
    alpha0: float,
) -> Tuple[float, float]:
    _, _, (tx, ty), (nx, ny) = base_curve.eval(0.0)
    target_tx, target_ty = (tx, ty) if side == 1 else (-tx, -ty)
    _, _, (twx, twy), _ = wheel_curve.eval(0.0)
    cos_a, sin_a = _rotation_from_to(twx, twy, target_tx, target_ty)
    cos_p = math.cos(alpha0)
    sin_p = math.sin(alpha0)
    pen_base = (d * (cos_p * nx + sin_p * tx), d * (cos_p * ny + sin_p * ty))
    inv_cos = cos_a
    inv_sin = -sin_a
    return _rotate(pen_base[0], pen_base[1], inv_cos, inv_sin)


def roll_pen_position(
    s: float,
    base_curve: BaseCurve,
    wheel_curve: BaseCurve,
    d: float,
    side: int,
    alpha0: float,
    epsilon: int = 1,
    pen_local: Tuple[float, float] | None = None,
) -> Tuple[float, float]:
    xb, yb, _, _ = base_curve.eval(s)
    if wheel_curve.length > 0:
        wheel_s = epsilon * s
        if wheel_curve.closed:
            wheel_s %= wheel_curve.length
        else:
            wheel_s = max(0.0, min(wheel_s, wheel_curve.length))
    else:
        wheel_s = 0.0
    xw, yw, _, _ = wheel_curve.eval(wheel_s)
    cos_a, sin_a = wheel_orientation(s, base_curve, wheel_curve, side, epsilon=epsilon)
    xw_rot, yw_rot = _rotate(xw, yw, cos_a, sin_a)
    cx = xb - xw_rot
    cy = yb - yw_rot
    if pen_local is None:
        pen_local = wheel_pen_local_vector(base_curve, wheel_curve, d, side, alpha0)
    px_local, py_local = pen_local
    px_rot, py_rot = _rotate(px_local, py_local, cos_a, sin_a)
    return cx + px_rot, cy + py_rot


@dataclass
class ModularTrackCurve(BaseCurve):
    segments: List[LineSegment | ArcSegment]
    closed: bool = False

    def __post_init__(self) -> None:
        total = sum(seg.length for seg in self.segments)
        BaseCurve.__init__(self, total, closed=self.closed)
        self._piecewise = PiecewiseCurve(self.segments, closed=self.closed)

    def eval(self, s: float) -> Tuple[float, float, Tuple[float, float], Tuple[float, float]]:
        return self._piecewise.eval(s)
