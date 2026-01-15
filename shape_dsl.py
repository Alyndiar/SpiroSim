from __future__ import annotations

from dataclasses import dataclass
import re
from typing import List, Optional, Union

_NUM_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"


@dataclass(frozen=True)
class CircleSpec:
    perimeter: float


@dataclass(frozen=True)
class RingSpec:
    inner: float
    outer: float


@dataclass(frozen=True)
class PolygonSpec:
    sides: int
    perimeter: float
    side_size: float
    corner_size: float


@dataclass(frozen=True)
class DropSpec:
    perimeter: float
    opposite: float
    half: float
    link: float


@dataclass(frozen=True)
class OblongSpec:
    perimeter: float
    cap_size: float


@dataclass(frozen=True)
class EllipseSpec:
    perimeter: float
    axis_a: float
    axis_b: float


AnalyticSpec = Union[CircleSpec, RingSpec, PolygonSpec, DropSpec, OblongSpec, EllipseSpec]


@dataclass(frozen=True)
class TrackOperator:
    kind: str
    jump_index: Optional[int] = None


@dataclass(frozen=True)
class ArcPiece:
    sweep_deg: float


@dataclass(frozen=True)
class StraightPiece:
    length: float


@dataclass(frozen=True)
class EndcapPiece:
    pass


@dataclass(frozen=True)
class IntersectionPiece:
    branches: int


TrackPiece = Union[ArcPiece, StraightPiece, EndcapPiece, IntersectionPiece]


@dataclass(frozen=True)
class TrackStep:
    operator: TrackOperator
    piece: TrackPiece


@dataclass(frozen=True)
class ModularTrackSpec:
    steps: List[TrackStep]


class DslParseError(ValueError):
    pass


def _clean_expr(expr: str) -> str:
    return "".join(ch for ch in expr if not ch.isspace())


def parse_analytic_expression(expr: str) -> AnalyticSpec:
    cleaned = _clean_expr(expr)
    if not cleaned:
        raise DslParseError("Empty expression")

    circle_match = re.fullmatch(rf"C\(({_NUM_RE})\)", cleaned)
    if circle_match:
        return CircleSpec(float(circle_match.group(1)))

    ring_match = re.fullmatch(rf"R\(({_NUM_RE}),({_NUM_RE})\)", cleaned)
    if ring_match:
        return RingSpec(float(ring_match.group(1)), float(ring_match.group(2)))

    poly_match = re.fullmatch(rf"P(\d+)\(({_NUM_RE}),({_NUM_RE})/({_NUM_RE})\)", cleaned)
    if poly_match:
        return PolygonSpec(
            int(poly_match.group(1)),
            float(poly_match.group(2)),
            float(poly_match.group(3)),
            float(poly_match.group(4)),
        )

    drop_match = re.fullmatch(rf"D\(({_NUM_RE}),({_NUM_RE})/({_NUM_RE})/({_NUM_RE})\)", cleaned)
    if drop_match:
        return DropSpec(
            float(drop_match.group(1)),
            float(drop_match.group(2)),
            float(drop_match.group(3)),
            float(drop_match.group(4)),
        )

    oblong_match = re.fullmatch(rf"O\(({_NUM_RE}),({_NUM_RE})\)", cleaned)
    if oblong_match:
        return OblongSpec(float(oblong_match.group(1)), float(oblong_match.group(2)))

    ellipse_match = re.fullmatch(rf"L\(({_NUM_RE}),({_NUM_RE})/({_NUM_RE})\)", cleaned)
    if ellipse_match:
        return EllipseSpec(
            float(ellipse_match.group(1)),
            float(ellipse_match.group(2)),
            float(ellipse_match.group(3)),
        )

    raise DslParseError(f"Unrecognized analytic DSL: {expr}")


def parse_modular_expression(expr: str) -> ModularTrackSpec:
    cleaned = _clean_expr(expr)
    if not cleaned:
        return ModularTrackSpec(steps=[])

    steps: List[TrackStep] = []
    idx = 0

    def _parse_number(start: int) -> tuple[Optional[float], int]:
        match = re.match(rf"{_NUM_RE}", cleaned[start:])
        if not match:
            return None, start
        number = float(match.group(0))
        return number, start + len(match.group(0))

    while idx < len(cleaned):
        op_char = "+"
        jump_index: Optional[int] = None
        if cleaned[idx] in "+-*":
            op_char = cleaned[idx]
            idx += 1
            if op_char == "*":
                jump_number, new_idx = _parse_number(idx)
                if jump_number is not None:
                    jump_index = int(jump_number)
                    idx = new_idx
        if idx >= len(cleaned):
            break

        if cleaned[idx] == "A":
            idx += 1
            if idx >= len(cleaned) or cleaned[idx] != "(":
                raise DslParseError("Expected '(' after A")
            idx += 1
            value, idx = _parse_number(idx)
            if value is None:
                raise DslParseError("Expected arc sweep value")
            if idx >= len(cleaned) or cleaned[idx] != ")":
                raise DslParseError("Expected ')' after arc value")
            idx += 1
            piece: TrackPiece = ArcPiece(value)
        elif cleaned[idx] == "S":
            idx += 1
            if idx >= len(cleaned) or cleaned[idx] != "(":
                raise DslParseError("Expected '(' after S")
            idx += 1
            value, idx = _parse_number(idx)
            if value is None:
                raise DslParseError("Expected straight length")
            if idx >= len(cleaned) or cleaned[idx] != ")":
                raise DslParseError("Expected ')' after straight length")
            idx += 1
            piece = StraightPiece(value)
        elif cleaned[idx] == "E":
            idx += 1
            piece = EndcapPiece()
        elif cleaned[idx] == "I":
            idx += 1
            if idx >= len(cleaned) or not cleaned[idx].isdigit():
                raise DslParseError("Expected intersection branch count")
            value, idx = _parse_number(idx)
            if value is None:
                raise DslParseError("Expected intersection branch count")
            piece = IntersectionPiece(int(value))
        else:
            raise DslParseError(f"Unexpected token at position {idx}: '{cleaned[idx]}'")

        steps.append(TrackStep(TrackOperator(op_char, jump_index), piece))

    return ModularTrackSpec(steps=steps)


def is_modular_expression(expr: str) -> bool:
    cleaned = _clean_expr(expr)
    return cleaned.startswith(("A", "S", "E", "I", "+", "-", "*"))
