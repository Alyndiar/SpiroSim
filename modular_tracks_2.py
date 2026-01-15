"""
Modular track engine based on the DSL defined in dsl_spec.md.
Implements arcs A(θ), straights S(L), endcaps E, and intersections I<n>.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

from shape_dsl import (
    DslParseError,
    ModularTrackSpec,
    TrackOperator,
    TrackStep,
    ArcPiece,
    StraightPiece,
    EndcapPiece,
    IntersectionPiece,
    parse_modular_expression,
)
from shape_geometry import ArcSegment, LineSegment, ModularTrackCurve, pen_position

Point = Tuple[float, float]


@dataclass
class ReferenceRing:
    inner_size: float = 144.0
    outer_size: float = 96.0

    def __post_init__(self) -> None:
        r_inner = self.inner_size / (2.0 * math.pi)
        r_outer = self.outer_size / (2.0 * math.pi)
        if r_inner > r_outer:
            self.inner_size, self.outer_size = self.outer_size, self.inner_size

    @property
    def r_inner(self) -> float:
        return self.inner_size / (2.0 * math.pi)

    @property
    def r_outer(self) -> float:
        return self.outer_size / (2.0 * math.pi)

    @property
    def width(self) -> float:
        return abs(self.r_outer - self.r_inner)

    @property
    def half_width(self) -> float:
        return 0.5 * self.width

    @property
    def r_center(self) -> float:
        return 0.5 * (self.r_outer + self.r_inner)


@dataclass
class TrackSegment:
    kind: str
    s_start: float
    s_end: float
    start: Point
    end: Point
    center: Optional[Point] = None
    radius: Optional[float] = None
    angle_start: Optional[float] = None
    angle_end: Optional[float] = None


@dataclass
class TrackBuildResult:
    segments: List[TrackSegment]
    points: List[Point]
    width: float
    half_width: float
    inner_length: float
    outer_length: float
    origin_offset: float
    origin_angle_offset: float
    ring: ReferenceRing

    @property
    def total_length(self) -> float:
        return self.segments[-1].s_end if self.segments else 0.0


@dataclass
class TrackRollContext:
    half_width: float
    r_wheel: float
    track_length: float
    sign_side: float
    wheel_size: int
    track_size: int


@dataclass
class TrackRollBundle:
    stylo: List[Point]
    centre: List[Point]
    contact: List[Point]
    marker0: List[Point]
    wheel_marker_indices: List[int]
    track_marker_indices: List[int]
    context: TrackRollContext


PIECES = frozenset({"a", "s", "e", "i"})


def _normalize(x: float, y: float) -> Tuple[float, float]:
    n = math.hypot(x, y)
    if n == 0:
        return (0.0, 0.0)
    return (x / n, y / n)


def _build_segments_from_spec(spec: ModularTrackSpec, ring: ReferenceRing) -> List[TrackSegment]:
    segments: List[TrackSegment] = []
    open_branches: List[Tuple[Point, float]] = []
    pos = (0.0, 0.0)
    heading = 0.0
    s_cursor = 0.0

    def add_segment(seg: LineSegment | ArcSegment) -> None:
        nonlocal s_cursor
        length = seg.length
        if isinstance(seg, LineSegment):
            segments.append(
                TrackSegment(
                    kind="line",
                    s_start=s_cursor,
                    s_end=s_cursor + length,
                    start=seg.start,
                    end=seg.end,
                )
            )
        else:
            segments.append(
                TrackSegment(
                    kind="arc",
                    s_start=s_cursor,
                    s_end=s_cursor + length,
                    start=(0.0, 0.0),
                    end=(0.0, 0.0),
                    center=seg.center,
                    radius=seg.radius,
                    angle_start=seg.angle_start,
                    angle_end=seg.angle_end,
                )
            )
        s_cursor += length

    for step in spec.steps:
        op = step.operator
        if op.kind == "*":
            if open_branches:
                idx = 0
                if op.jump_index is not None:
                    idx = max(0, min(int(op.jump_index), len(open_branches) - 1))
                pos, heading = open_branches.pop(idx)
            continue
        reverse = op.kind == "-"
        if reverse:
            heading = heading + math.pi

        if isinstance(step.piece, ArcPiece):
            sweep = math.radians(step.piece.sweep_deg)
            if reverse:
                sweep = -sweep
            left_x = -math.sin(heading)
            left_y = math.cos(heading)
            sign = 1.0 if sweep >= 0 else -1.0
            center = (
                pos[0] + left_x * ring.r_center * sign,
                pos[1] + left_y * ring.r_center * sign,
            )
            angle_start = math.atan2(pos[1] - center[1], pos[0] - center[0])
            angle_end = angle_start + sweep
            arc = ArcSegment(center, ring.r_center, angle_start, angle_end)
            add_segment(arc)
            pos = (
                center[0] + ring.r_center * math.cos(angle_end),
                center[1] + ring.r_center * math.sin(angle_end),
            )
            heading += sweep
        elif isinstance(step.piece, StraightPiece):
            length = step.piece.length
            if length < 0:
                length = 0.0
            start = pos
            end = (
                pos[0] + length * math.cos(heading),
                pos[1] + length * math.sin(heading),
            )
            add_segment(LineSegment(start, end))
            pos = end
        elif isinstance(step.piece, EndcapPiece):
            sweep = math.pi
            left_x = -math.sin(heading)
            left_y = math.cos(heading)
            center = (
                pos[0] + left_x * ring.half_width,
                pos[1] + left_y * ring.half_width,
            )
            angle_start = math.atan2(pos[1] - center[1], pos[0] - center[0])
            angle_end = angle_start + sweep
            arc = ArcSegment(center, ring.half_width, angle_start, angle_end)
            add_segment(arc)
            pos = (
                center[0] + ring.half_width * math.cos(angle_end),
                center[1] + ring.half_width * math.sin(angle_end),
            )
            heading += sweep
        elif isinstance(step.piece, IntersectionPiece):
            branches = max(3, step.piece.branches)
            for i in range(1, branches):
                open_branches.append((pos, heading + (2.0 * math.pi * i / branches)))
        if reverse:
            heading = heading + math.pi

    return segments


def _sample_segments(segments: List[TrackSegment], samples: int) -> List[Point]:
    if not segments:
        return [(0.0, 0.0)]
    points: List[Point] = []
    total = segments[-1].s_end
    for i in range(samples):
        s = total * i / max(1, samples - 1)
        (x, y), _, _ = _interpolate_on_segments(s, segments)
        points.append((x, y))
    return points


def _interpolate_on_segments(s: float, segments: List[TrackSegment]) -> Tuple[Point, float, Point]:
    if not segments:
        return (0.0, 0.0), 0.0, (0.0, 1.0)
    total = segments[-1].s_end
    s = max(0.0, min(s, total))
    seg = segments[-1]
    for candidate in segments:
        if candidate.s_start <= s <= candidate.s_end:
            seg = candidate
            break
    local_s = s - seg.s_start
    if seg.kind == "line":
        length = seg.s_end - seg.s_start
        t = 0.0 if length == 0 else local_s / length
        x = seg.start[0] + (seg.end[0] - seg.start[0]) * t
        y = seg.start[1] + (seg.end[1] - seg.start[1]) * t
        tx, ty = _normalize(seg.end[0] - seg.start[0], seg.end[1] - seg.start[1])
        nx, ny = -ty, tx
        return (x, y), math.atan2(ty, tx), (nx, ny)
    if seg.kind == "arc" and seg.center is not None and seg.radius is not None:
        length = seg.s_end - seg.s_start
        t = 0.0 if length == 0 else local_s / length
        angle_start = seg.angle_start if seg.angle_start is not None else 0.0
        angle_end = seg.angle_end if seg.angle_end is not None else angle_start
        angle = angle_start + (angle_end - angle_start) * t
        x = seg.center[0] + seg.radius * math.cos(angle)
        y = seg.center[1] + seg.radius * math.sin(angle)
        delta = angle_end - angle_start
        sign = 1.0 if delta >= 0 else -1.0
        tx, ty = _normalize(-math.sin(angle) * sign, math.cos(angle) * sign)
        nx, ny = -ty, tx
        return (x, y), math.atan2(ty, tx), (nx, ny)
    return (0.0, 0.0), 0.0, (0.0, 1.0)


def build_track_from_notation(
    notation: str,
    inner_size: float = 96.0,
    outer_size: float = 144.0,
    steps_per_unit: int = 3,
) -> TrackBuildResult:
    ring = ReferenceRing(inner_size, outer_size)
    spec = parse_modular_expression(notation)
    segments = _build_segments_from_spec(spec, ring)
    points = _sample_segments(segments, max(2, int(steps_per_unit * max(1.0, sum(seg.s_end - seg.s_start for seg in segments)))))
    inner_length, mid_length, outer_length = compute_track_lengths(segments, half_width=ring.half_width)
    return TrackBuildResult(
        segments=segments,
        points=points,
        width=ring.width,
        half_width=ring.half_width,
        inner_length=inner_length,
        outer_length=outer_length,
        origin_offset=0.0,
        origin_angle_offset=0.0,
        ring=ring,
    )


def compute_track_lengths(
    track_or_segments,
    inner_size: Optional[float] = None,
    outer_size: Optional[float] = None,
    half_width: Optional[float] = None,
) -> Tuple[float, float, float]:
    if isinstance(track_or_segments, TrackBuildResult):
        segments = track_or_segments.segments
        half_w = track_or_segments.half_width
        total_length = track_or_segments.total_length
    else:
        segments = track_or_segments
        if half_width is not None:
            half_w = half_width
        elif inner_size is not None and outer_size is not None:
            ring = ReferenceRing(inner_size, outer_size)
            half_w = ring.half_width
        else:
            half_w = 0.0
        total_length = segments[-1].s_end if segments else 0.0

    inner_length = 0.0
    outer_length = 0.0
    for seg in segments:
        if seg.kind == "line":
            length = seg.s_end - seg.s_start
            inner_length += length
            outer_length += length
        elif seg.kind == "arc" and seg.radius is not None and seg.angle_start is not None and seg.angle_end is not None:
            delta = seg.angle_end - seg.angle_start
            inner_length += abs((seg.radius - half_w) * delta)
            outer_length += abs((seg.radius + half_w) * delta)
    return inner_length, total_length, outer_length


def compute_track_polylines(
    track: TrackBuildResult,
    samples: int = 800,
    half_width: Optional[float] = None,
) -> Tuple[List[Point], List[Point], List[Point], float]:
    center = _sample_segments(track.segments, samples)
    offset = track.half_width if half_width is None else half_width
    inner: List[Point] = []
    outer: List[Point] = []
    for i in range(len(center) - 1):
        x0, y0 = center[i]
        x1, y1 = center[i + 1]
        dx = x1 - x0
        dy = y1 - y0
        nx, ny = _normalize(-dy, dx)
        inner.append((x0 - nx * offset, y0 - ny * offset))
        outer.append((x0 + nx * offset, y0 + ny * offset))
    if center:
        inner.append(inner[-1])
        outer.append(outer[-1])
    return center, inner, outer, offset


def build_track_and_bundle_from_notation(
    notation: str,
    wheel_size: float,
    hole_offset: float,
    steps: int,
    relation: str,
    phase_offset: float,
    inner_size: float,
    outer_size: float,
) -> Tuple[TrackBuildResult, TrackRollBundle]:
    track = build_track_from_notation(notation, inner_size=inner_size, outer_size=outer_size)
    base_curve = ModularTrackCurve([
        LineSegment(seg.start, seg.end) if seg.kind == "line" else ArcSegment(
            seg.center,
            seg.radius or 0.0,
            seg.angle_start or 0.0,
            seg.angle_end or 0.0,
        )
        for seg in track.segments
    ], closed=False)
    r = wheel_size / (2.0 * math.pi) if wheel_size else 1.0
    d = max(0.0, (wheel_size / (2.0 * math.pi)) - hole_offset)
    side = 1 if relation == "dedans" else -1
    epsilon = -side
    s_max = track.total_length
    stylo: List[Point] = []
    centre: List[Point] = []
    contact: List[Point] = []
    marker0: List[Point] = []
    wheel_marker_indices: List[int] = []
    track_marker_indices: List[int] = []
    for i in range(steps):
        s = s_max * i / max(1, steps - 1)
        px, py = pen_position(s, base_curve, r, d, side, alpha0=0.0, epsilon=epsilon)
        stylo.append((px, py))
        xb, yb, (tx, ty), (nx, ny) = base_curve.eval(s)
        centre.append((xb + side * r * nx, yb + side * r * ny))
        contact.append((xb, yb))
        marker0.append((xb, yb))
        wheel_marker_indices.append(i)
        track_marker_indices.append(i)
    context = TrackRollContext(
        half_width=track.half_width,
        r_wheel=r,
        track_length=track.total_length,
        sign_side=side,
        wheel_size=int(round(wheel_size)),
        track_size=int(round(track.total_length)),
    )
    bundle = TrackRollBundle(
        stylo=stylo,
        centre=centre,
        contact=contact,
        marker0=marker0,
        wheel_marker_indices=wheel_marker_indices,
        track_marker_indices=track_marker_indices,
        context=context,
    )
    return track, bundle


def build_track_and_bundle_from_spec(
    spec: ModularTrackSpec,
    ring: ReferenceRing,
    wheel_size: float,
    hole_offset: float,
    steps: int,
    relation: str,
) -> Tuple[TrackBuildResult, TrackRollBundle]:
    notation = ""
    return build_track_and_bundle_from_notation(
        notation,
        wheel_size=wheel_size,
        hole_offset=hole_offset,
        steps=steps,
        relation=relation,
        phase_offset=0.0,
        inner_size=ring.inner_size,
        outer_size=ring.outer_size,
    )


def split_valid_modular_notation(text: str) -> Tuple[str, str, bool]:
    cleaned = "".join(ch for ch in text if not ch.isspace())
    if not cleaned:
        return "", "", False
    idx = 0
    last_valid = 0
    has_piece = False
    while idx < len(cleaned):
        if cleaned[idx] in "+-*":
            idx += 1
            while idx < len(cleaned) and cleaned[idx].isdigit():
                idx += 1
        if idx >= len(cleaned):
            break
        if cleaned[idx] in {"A", "S"}:
            idx += 1
            if idx >= len(cleaned) or cleaned[idx] != "(":
                break
            idx += 1
            while idx < len(cleaned) and (cleaned[idx].isdigit() or cleaned[idx] == "." or cleaned[idx] in "+-"):
                idx += 1
            if idx >= len(cleaned) or cleaned[idx] != ")":
                break
            idx += 1
            has_piece = True
        elif cleaned[idx] == "E":
            idx += 1
            has_piece = True
        elif cleaned[idx] == "I":
            idx += 1
            while idx < len(cleaned) and cleaned[idx].isdigit():
                idx += 1
            has_piece = True
        else:
            break
        last_valid = idx
    return cleaned[:last_valid], cleaned[last_valid:], has_piece


def parse_track_notation(text: str) -> ModularTrackSpec:
    try:
        return parse_modular_expression(text)
    except DslParseError:
        return ModularTrackSpec(steps=[])


def compute_track_polylines_for_notation(
    notation: str,
    inner_size: float,
    outer_size: float,
    samples: int = 800,
) -> Tuple[List[Point], List[Point], List[Point], float]:
    track = build_track_from_notation(notation, inner_size=inner_size, outer_size=outer_size)
    return compute_track_polylines(track, samples=samples)
