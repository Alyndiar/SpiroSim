from __future__ import annotations

from typing import Optional
import functools


import modular_tracks
from shape_geometry import ArcSegment, LineSegment, ModularTrackCurve
from shape_geometry import (
    BaseCurve,
    EllipseCurve,
    build_circle,
    build_drop,
    build_oblong,
    build_rounded_polygon,
)
from shape_rsdl import (
    CircleSpec,
    DropSpec,
    EllipseSpec,
    OblongSpec,
    PolygonSpec,
    RingSpec,
    RsdlParseError,
    is_modular_expression,
    normalize_rsdl_text,
    parse_analytic_expression,
    parse_modular_expression,
)


@functools.lru_cache(maxsize=256)
def _parse_analytic_cached(expression: str):
    return parse_analytic_expression(expression)


@functools.lru_cache(maxsize=256)
def curve_from_expression(expression: str, relation: str) -> Optional[BaseCurve]:
    spec = _parse_analytic_cached(expression)
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


def is_polygon_expression(expression: str) -> bool:
    try:
        spec = _parse_analytic_cached(expression)
    except RsdlParseError:
        return False
    return spec.__class__.__name__ == "PolygonSpec"



@functools.lru_cache(maxsize=128)
def _build_modular_curve_cached(
    notation: str,
    inner_size: int,
    outer_size: int,
    steps_per_unit: int,
) -> ModularTrackCurve:
    track = modular_tracks.build_track_from_notation(
        notation,
        inner_size=inner_size,
        outer_size=outer_size,
        steps_per_unit=steps_per_unit,
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


def build_modular_curve(
    notation: str,
    inner_size: int,
    outer_size: int,
    steps_per_unit: int = 3,
) -> ModularTrackCurve:
    return _build_modular_curve_cached(
        notation,
        inner_size,
        outer_size,
        steps_per_unit,
    )


__all__ = [
    "CircleSpec",
    "DropSpec",
    "EllipseSpec",
    "OblongSpec",
    "PolygonSpec",
    "RingSpec",
    "RsdlParseError",
    "curve_from_expression",
    "build_modular_curve",
    "is_modular_expression",
    "is_polygon_expression",
    "normalize_rsdl_text",
    "parse_analytic_expression",
    "parse_modular_expression",
]
