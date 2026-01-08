"""
Nouvelle génération de pistes modulaires basée sur une notation algébrique
simplifiée. La totalité de la logique précédente est remplacée afin de coller
au cahier des charges suivant :

- Une piste est décrite par une suite de blocs "lettre+nombre" séparés par les
  opérateurs "+", "-" ou "*". Un signe initial optionnel précise le côté vers
  lequel tourne la première pièce.
- Les paramètres de référence (dents intérieures / extérieures et pas) sont
  fournis par l'appelant, par défaut 144 dents intérieures, 96 dents
  extérieures et le pas standard du Spirograph.
- La largeur de piste est la différence de rayon entre les deux cercles de
  référence ; elle est donc constante pour toutes les pièces.
- Les pièces reconnues sont :
    * ``aNN`` : arc de NN degrés ; l'orientation est donnée par l'opérateur
      précédent (``+`` pour tourner à gauche, ``-`` pour tourner à droite).
    * ``dNN`` : droite de NN dents.
    * ``b``   : bout arrondi (demi-cercle) reliant les deux côtés de la piste ;
      aucune longueur n'est indiquée après la lettre.
    * ``y``   : jonction triangulaire composée de trois arcs de 120° espacés de
      la largeur de la piste. Les dents de ces arcs sont toujours évaluées avec
      le nombre de dents intérieures de l'anneau de référence.
    * ``nNN`` : décalage de l'origine en nombre de dents, appliqué dans la
      direction indiquée par le signe précédent (gauche = ``+``, droite = ``-``).
    * ``oNN`` : décalage angulaire de l'origine en degrés, avec le même
      convention de signe que ``n``.
- L'opérateur ``*`` fait passer à la prochaine branche ouverte (issue d'un
  ``y`` ou d'un bout ``b`` qui laisse un côté libre).
- À l'ajout de chaque pièce, les dents consommées sur les pistes gauche et
  droite sont comptabilisées. Une fois la piste terminée, la piste la plus
  courte est considérée comme l'intérieur ; la plus longue, comme l'extérieur.

Ce module reste indépendant de Qt et expose des primitives de construction de
piste, d'interpolation sur la géométrie et une génération de courbe trochoïdale
compatible avec les appels existants.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Tuple

Point = Tuple[float, float]


# ---------------------------------------------------------------------------
# 1) Structures de base
# ---------------------------------------------------------------------------

@dataclass
class ReferenceRing:
    """Paramétrage de l'anneau de référence.

    Les valeurs sont automatiquement réordonnées pour que ``r_inner`` soit
    toujours le plus petit rayon, conformément à la notion de "rayon intérieur".
    """

    inner_teeth: float = 144.0
    outer_teeth: float = 96.0
    pitch_mm_per_tooth: float = 0.65

    def __post_init__(self) -> None:
        if self.pitch_mm_per_tooth <= 0:
            raise ValueError("pitch_mm_per_tooth doit être positif")
        r_inner = (self.inner_teeth * self.pitch_mm_per_tooth) / (2.0 * math.pi)
        r_outer = (self.outer_teeth * self.pitch_mm_per_tooth) / (2.0 * math.pi)
        if r_inner > r_outer:
            # on réordonne pour garantir r_inner <= r_outer
            self.inner_teeth, self.outer_teeth = self.outer_teeth, self.inner_teeth

    @property
    def r_inner(self) -> float:
        return (self.inner_teeth * self.pitch_mm_per_tooth) / (2.0 * math.pi)

    @property
    def r_outer(self) -> float:
        return (self.outer_teeth * self.pitch_mm_per_tooth) / (2.0 * math.pi)

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
class TrackBlock:
    """Bloc issu de la notation algébrique."""

    operator: str  # '+', '-', '*'
    kind: str      # 'a', 'd', 'b', 'y', 'n', 'o', 'branch'
    value: Optional[float] = None
    raw: str = ""


@dataclass
class ParsedTrack:
    blocks: List[TrackBlock]
    origin_teeth_offset: float
    origin_angle_offset: float
    default_turn: int


@dataclass
class TrackSegment:
    """Segment de la piste (médiane ou côté reparamétré)."""

    kind: str  # 'line' ou 'arc'
    s_start: float
    s_end: float
    turn: int  # +1 gauche, -1 droite, 0 pour les droites
    start: Point
    end: Point
    center: Optional[Point] = None
    radius: Optional[float] = None
    angle_start: Optional[float] = None
    angle_end: Optional[float] = None
    left_is_inner: Optional[bool] = None
    length_left: float = 0.0
    length_right: float = 0.0


@dataclass
class TrackBuildResult:
    segments: List[TrackSegment]
    points: List[Point]
    width: float
    half_width: float
    left_teeth: float
    right_teeth: float
    inner_teeth: float
    outer_teeth: float
    inner_side: str
    origin_teeth_offset: float
    origin_angle_offset: float
    ring: ReferenceRing

    @property
    def total_length(self) -> float:
        return self.segments[-1].s_end if self.segments else 0.0


@dataclass
class TrackRollBundle:
    stylo: List[Point]
    centre: List[Point]
    contact: List[Point]
    marker0: List[Point]
    wheel_teeth_indices: List[int]
    track_teeth_indices: List[int]
    context: "TrackRollContext"


@dataclass
class TrackRollContext:
    half_width: float
    r_wheel: float
    N_wheel: int
    N_track: int
    track_length: float
    sign_side: float
    pitch_mm_per_tooth: float


# ---------------------------------------------------------------------------
# 2) Parsing de la notation
# ---------------------------------------------------------------------------

_VALID_PIECES = {"a", "d", "b", "y", "n", "o"}


def _parse_number(s: str, idx: int) -> Tuple[Optional[float], int]:
    start = idx
    while idx < len(s) and (s[idx].isdigit() or s[idx] == "."):
        idx += 1
    if idx == start:
        return None, idx
    return float(s[start:idx]), idx


def parse_track_notation(text: str) -> ParsedTrack:
    """Parse la notation algébrique décrite dans le cahier des charges."""

    cleaned = text.replace(" ", "").lower()
    if not cleaned:
        return ParsedTrack(blocks=[], origin_teeth_offset=0.0, origin_angle_offset=0.0, default_turn=1)

    blocks: List[TrackBlock] = []
    idx = 0
    n = len(cleaned)
    pending_op: Optional[str] = None

    def _consume_operator(pos: int) -> Tuple[Optional[str], int]:
        if pos < n and cleaned[pos] in "+-*":
            return cleaned[pos], pos + 1
        return None, pos

    default_turn = 1
    op, idx = _consume_operator(idx)
    if op in {"+", "-"}:
        default_turn = 1 if op == "+" else -1
        pending_op = op
    elif op == "*":
        blocks.append(TrackBlock(operator="*", kind="branch", raw="*"))
        pending_op = None
    else:
        pending_op = "+"

    origin_teeth_offset = 0.0
    origin_angle_offset = 0.0

    while idx < n:
        op, idx = _consume_operator(idx)
        if op is not None:
            if op == "*":
                blocks.append(TrackBlock(operator="*", kind="branch", raw="*"))
                pending_op = None
                continue
            pending_op = op
        if idx >= n:
            break

        letter = cleaned[idx]
        idx += 1
        if letter not in _VALID_PIECES:
            raise ValueError(f"Pièce inconnue : {letter}")

        value: Optional[float] = None
        if letter in {"a", "d", "n", "o"}:
            value, idx = _parse_number(cleaned, idx)
            if value is None:
                raise ValueError(f"La pièce '{letter}' doit être suivie d'un nombre")
        elif letter in {"b", "y"}:
            value = None

        operator = pending_op or "+"
        raw = f"{operator}{letter}{'' if value is None else value}"
        block = TrackBlock(operator=operator, kind=letter, value=value, raw=raw)
        blocks.append(block)

        if letter == "n":
            delta = value or 0.0
            origin_teeth_offset += delta if operator == "+" else -delta
        if letter == "o":
            delta = value or 0.0
            origin_angle_offset += delta if operator == "+" else -delta

    return ParsedTrack(
        blocks=blocks,
        origin_teeth_offset=origin_teeth_offset,
        origin_angle_offset=origin_angle_offset,
        default_turn=default_turn,
    )


# ---------------------------------------------------------------------------
# 3) Construction géométrique
# ---------------------------------------------------------------------------

@dataclass
class _Pose:
    point: Point
    tangent: Point
    normal: Point


def _normalize(vx: float, vy: float) -> Point:
    n = math.hypot(vx, vy)
    if n == 0:
        return 0.0, 0.0
    return vx / n, vy / n


def _rotate(vec: Point, angle: float) -> Point:
    vx, vy = vec
    c = math.cos(angle)
    s = math.sin(angle)
    return vx * c - vy * s, vx * s + vy * c


def _append_line(
    pose: _Pose,
    length_mm: float,
    s_cursor: float,
) -> Tuple[TrackSegment, _Pose, float, float, float]:
    if length_mm < 0:
        length_mm = 0.0
    x0, y0 = pose.point
    tx, ty = pose.tangent
    nx, ny = pose.normal

    x1 = x0 + tx * length_mm
    y1 = y0 + ty * length_mm
    segment = TrackSegment(
        kind="line",
        s_start=s_cursor,
        s_end=s_cursor + length_mm,
        turn=0,
        start=(x0, y0),
        end=(x1, y1),
        left_is_inner=None,
        length_left=length_mm,
        length_right=length_mm,
    )
    new_pose = _Pose(point=(x1, y1), tangent=(tx, ty), normal=(nx, ny))
    return segment, new_pose, length_mm, length_mm, s_cursor + length_mm


def _append_arc(
    pose: _Pose,
    angle_deg: float,
    turn_sign: int,
    ring: ReferenceRing,
    s_cursor: float,
    radius_override: Optional[float] = None,
    teeth_for_both_sides: Optional[float] = None,
) -> Tuple[TrackSegment, _Pose, float, float, float]:
    angle_rad = math.radians(angle_deg)
    if angle_rad == 0:
        empty_segment = TrackSegment(
            kind="arc",
            s_start=s_cursor,
            s_end=s_cursor,
            turn=turn_sign,
            start=pose.point,
            end=pose.point,
            center=pose.point,
            radius=0.0,
            angle_start=0.0,
            angle_end=0.0,
            left_is_inner=turn_sign > 0,
            length_left=0.0,
            length_right=0.0,
        )
        return empty_segment, pose, 0.0, 0.0, s_cursor

    r_center = radius_override if radius_override is not None else ring.r_center
    half_w = ring.half_width
    left_is_inner = turn_sign > 0
    radius_left = r_center - turn_sign * (+1) * half_w
    radius_right = r_center - turn_sign * (-1) * half_w

    # Longueurs géométriques
    length_center = abs(r_center * angle_rad)
    length_left = abs(radius_left * angle_rad)
    length_right = abs(radius_right * angle_rad)

    if teeth_for_both_sides is not None:
        # Imposer les longueurs via la règle des dents explicitée
        length_left = length_right = teeth_for_both_sides * ring.pitch_mm_per_tooth
        length_center = length_left  # cohérence locale

    x0, y0 = pose.point
    tx, ty = pose.tangent
    nx, ny = pose.normal

    cx = x0 + turn_sign * nx * r_center
    cy = y0 + turn_sign * ny * r_center
    center = (cx, cy)
    angle_start = math.atan2(y0 - cy, x0 - cx)
    angle_end = angle_start + turn_sign * angle_rad

    end_point = (cx + r_center * math.cos(angle_end), cy + r_center * math.sin(angle_end))
    new_tangent = _rotate((tx, ty), turn_sign * angle_rad)
    new_tangent = _normalize(*new_tangent)
    new_normal = (-new_tangent[1], new_tangent[0])
    new_pose = _Pose(point=end_point, tangent=new_tangent, normal=new_normal)

    segment = TrackSegment(
        kind="arc",
        s_start=s_cursor,
        s_end=s_cursor + length_center,
        turn=turn_sign,
        start=(x0, y0),
        end=end_point,
        center=center,
        radius=r_center,
        angle_start=angle_start,
        angle_end=angle_end,
        left_is_inner=left_is_inner,
        length_left=length_left,
        length_right=length_right,
    )

    return segment, new_pose, length_left, length_right, s_cursor + length_center


@dataclass
class _BranchState:
    pose: _Pose
    pending_segments: List[TrackSegment] = field(default_factory=list)


def _apply_pending_segments(
    branch: _BranchState,
    segments: List[TrackSegment],
    s_cursor: float,
) -> float:
    for seg in branch.pending_segments:
        length = seg.s_end - seg.s_start
        seg.s_start = s_cursor
        seg.s_end = s_cursor + length
        segments.append(seg)
        s_cursor += length
        branch.pose = _Pose(point=seg.end, tangent=_segment_tangent(seg), normal=_segment_normal(seg))
    branch.pending_segments.clear()
    return s_cursor


def _segment_tangent(seg: TrackSegment) -> Point:
    if seg.kind == "line":
        dx = seg.end[0] - seg.start[0]
        dy = seg.end[1] - seg.start[1]
        return _normalize(dx, dy)
    if seg.kind == "arc" and seg.center is not None and seg.angle_end is not None:
        tx, ty = _normalize(-math.sin(seg.angle_end), math.cos(seg.angle_end))
        if seg.turn < 0:
            tx, ty = -tx, -ty
        return tx, ty
    return (1.0, 0.0)


def _segment_normal(seg: TrackSegment) -> Point:
    tx, ty = _segment_tangent(seg)
    return -ty, tx


def _build_segments(parsed: ParsedTrack, ring: ReferenceRing) -> TrackBuildResult:
    pose = _Pose(point=(0.0, 0.0), tangent=(1.0, 0.0), normal=(0.0, 1.0))
    s_cursor = 0.0
    left_teeth = 0.0
    right_teeth = 0.0
    segments: List[TrackSegment] = []
    branch_queue: List[_BranchState] = []

    for block in parsed.blocks:
        if block.kind == "branch":
            if branch_queue:
                next_branch = branch_queue.pop(0)
                s_cursor = _apply_pending_segments(next_branch, segments, s_cursor)
                pose = next_branch.pose
            continue

        if block.kind in {"n", "o"}:
            # Les décalages d'origine sont déjà accumulés lors du parsing
            continue

        turn_sign = 1 if block.operator == "+" else -1

        if block.kind == "d":
            length_mm = (block.value or 0.0) * ring.pitch_mm_per_tooth
            seg, pose, len_left, len_right, s_cursor = _append_line(pose, length_mm, s_cursor)
            segments.append(seg)
            left_teeth += block.value or 0.0
            right_teeth += block.value or 0.0
            continue

        if block.kind == "a":
            seg, pose, len_left, len_right, s_cursor = _append_arc(
                pose,
                angle_deg=block.value or 0.0,
                turn_sign=turn_sign,
                ring=ring,
                s_cursor=s_cursor,
            )
            segments.append(seg)
            # comptage des dents basé sur la position intérieure/extérieure
            teeth_inner = ring.inner_teeth * ((block.value or 0.0) / 360.0)
            teeth_outer = ring.outer_teeth * ((block.value or 0.0) / 360.0)
            if turn_sign > 0:
                left_teeth += teeth_inner
                right_teeth += teeth_outer
            else:
                left_teeth += teeth_outer
                right_teeth += teeth_inner
            continue

        if block.kind == "b":
            seg, pose, len_left, len_right, s_cursor = _append_arc(
                pose,
                angle_deg=180.0,
                turn_sign=turn_sign,
                ring=ring,
                s_cursor=s_cursor,
                radius_override=ring.half_width,
            )
            segments.append(seg)
            teeth_here = (seg.s_end - seg.s_start) / ring.pitch_mm_per_tooth
            left_teeth += teeth_here
            right_teeth += teeth_here
            # Le bout ouvre implicitement une branche libre
            branch_queue.append(_BranchState(pose=_Pose(point=pose.point, tangent=pose.tangent, normal=pose.normal)))
            continue

        if block.kind == "y":
            angle = 120.0
            r_geo = ring.width / math.sqrt(3)
            # Arc principal (branche courante)
            seg0, new_pose, len_left, len_right, s_cursor = _append_arc(
                pose,
                angle_deg=angle,
                turn_sign=turn_sign,
                ring=ring,
                s_cursor=s_cursor,
                radius_override=r_geo,
                teeth_for_both_sides=ring.inner_teeth * (angle / 360.0),
            )
            segments.append(seg0)
            left_teeth += ring.inner_teeth * (angle / 360.0)
            right_teeth += ring.inner_teeth * (angle / 360.0)

            heading = math.atan2(pose.tangent[1], pose.tangent[0])
            alpha0 = heading - turn_sign * (math.pi / 2.0)
            hub_x = pose.point[0] - r_geo * math.cos(alpha0)
            hub_y = pose.point[1] - r_geo * math.sin(alpha0)
            hub = (hub_x, hub_y)

            # Préparer deux autres branches
            for k in (1, 2):
                alpha_k = alpha0 + k * (2.0 * math.pi / 3.0)
                start_pt = (hub_x + r_geo * math.cos(alpha_k), hub_y + r_geo * math.sin(alpha_k))
                start_tan = _rotate(pose.tangent, k * (2.0 * math.pi / 3.0))
                start_tan = _normalize(*start_tan)
                start_norm = (-start_tan[1], start_tan[0])
                branch_pose = _Pose(point=start_pt, tangent=start_tan, normal=start_norm)

                seg_branch, branch_end_pose, _, _, _ = _append_arc(
                    branch_pose,
                    angle_deg=angle,
                    turn_sign=turn_sign,
                    ring=ring,
                    s_cursor=0.0,
                    radius_override=r_geo,
                    teeth_for_both_sides=ring.inner_teeth * (angle / 360.0),
                )
                branch_state = _BranchState(pose=branch_end_pose, pending_segments=[seg_branch])
                branch_queue.append(branch_state)

            pose = new_pose
            continue

    # Appliquer les décalages d'origine sur la géométrie
    segments = _apply_origin_offsets(
        segments,
        rotation_deg=parsed.origin_angle_offset,
        offset_teeth=parsed.origin_teeth_offset,
        ring=ring,
    )

    points = list(_sample_segments(segments, max(200, len(segments) * 10)))

    inner_teeth = min(left_teeth, right_teeth)
    outer_teeth = max(left_teeth, right_teeth)
    if math.isclose(inner_teeth, outer_teeth, rel_tol=1e-9, abs_tol=1e-9):
        inner_side = "both"
    elif left_teeth <= right_teeth:
        inner_side = "left"
    else:
        inner_side = "right"

    return TrackBuildResult(
        segments=segments,
        points=points,
        width=ring.width,
        half_width=ring.half_width,
        left_teeth=left_teeth,
        right_teeth=right_teeth,
        inner_teeth=inner_teeth,
        outer_teeth=outer_teeth,
        inner_side=inner_side,
        origin_teeth_offset=parsed.origin_teeth_offset,
        origin_angle_offset=parsed.origin_angle_offset,
        ring=ring,
    )


def _apply_origin_offsets(
    segments: List[TrackSegment],
    *,
    rotation_deg: float,
    offset_teeth: float,
    ring: ReferenceRing,
) -> List[TrackSegment]:
    if not segments:
        return []

    rot = math.radians(rotation_deg)
    cos_r = math.cos(rot)
    sin_r = math.sin(rot)
    offset_mm = offset_teeth * ring.pitch_mm_per_tooth
    # translation suivant la normale initiale (0, 1)
    dx = 0.0
    dy = offset_mm

    def _transform(p: Point) -> Point:
        x, y = p
        x, y = x * cos_r - y * sin_r, x * sin_r + y * cos_r
        return x + dx, y + dy

    transformed: List[TrackSegment] = []
    for seg in segments:
        start = _transform(seg.start)
        end = _transform(seg.end)
        center = _transform(seg.center) if seg.center is not None else None
        transformed.append(
            TrackSegment(
                kind=seg.kind,
                s_start=seg.s_start,
                s_end=seg.s_end,
                turn=seg.turn,
                start=start,
                end=end,
                center=center,
                radius=seg.radius,
                angle_start=None if seg.angle_start is None else seg.angle_start + rot,
                angle_end=None if seg.angle_end is None else seg.angle_end + rot,
                left_is_inner=seg.left_is_inner,
                length_left=seg.length_left,
                length_right=seg.length_right,
            )
        )

    return transformed


def _sample_segments(segments: List[TrackSegment], samples: int) -> Iterable[Point]:
    if not segments:
        return [(0.0, 0.0)]

    for seg in segments:
        if seg.s_end <= seg.s_start:
            continue
        n_steps = max(2, samples // max(1, len(segments)))
        if seg.kind == "line":
            for i in range(n_steps):
                t = i / float(n_steps - 1)
                x = seg.start[0] + (seg.end[0] - seg.start[0]) * t
                y = seg.start[1] + (seg.end[1] - seg.start[1]) * t
                yield (x, y)
        elif seg.kind == "arc" and seg.center is not None and seg.radius is not None:
            angle0 = seg.angle_start or 0.0
            angle1 = seg.angle_end or angle0
            for i in range(n_steps):
                t = i / float(n_steps - 1)
                a = angle0 + (angle1 - angle0) * t
                x = seg.center[0] + seg.radius * math.cos(a)
                y = seg.center[1] + seg.radius * math.sin(a)
                yield (x, y)


# ---------------------------------------------------------------------------
# 4) Interpolation sur la géométrie
# ---------------------------------------------------------------------------

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
        angle_start = seg.angle_start or 0.0
        angle_end = seg.angle_end or angle_start
        a = angle_start + (angle_end - angle_start) * t
        x = seg.center[0] + seg.radius * math.cos(a)
        y = seg.center[1] + seg.radius * math.sin(a)
        tangent = (-math.sin(a), math.cos(a))
        if seg.turn < 0:
            tangent = (-tangent[0], -tangent[1])
        tx, ty = _normalize(*tangent)
        nx, ny = -ty, tx
        return (x, y), math.atan2(ty, tx), (nx, ny)

    return (0.0, 0.0), 0.0, (0.0, 1.0)


def _make_side_segments(track: TrackBuildResult, side: str) -> List[TrackSegment]:
    side_sign = 1 if side == "left" else -1
    half_w = track.half_width
    remapped: List[TrackSegment] = []
    s_cursor = 0.0

    for seg in track.segments:
        if seg.kind == "line":
            tx, ty = _normalize(seg.end[0] - seg.start[0], seg.end[1] - seg.start[1])
            nx, ny = -ty, tx
            offset_x = side_sign * half_w * nx
            offset_y = side_sign * half_w * ny
            start = (seg.start[0] + offset_x, seg.start[1] + offset_y)
            end = (seg.end[0] + offset_x, seg.end[1] + offset_y)
            length = math.hypot(end[0] - start[0], end[1] - start[1])
            remapped.append(
                TrackSegment(
                    kind="line",
                    s_start=s_cursor,
                    s_end=s_cursor + length,
                    turn=seg.turn,
                    start=start,
                    end=end,
                    left_is_inner=seg.left_is_inner,
                    length_left=length,
                    length_right=length,
                )
            )
            s_cursor += length
        elif seg.kind == "arc" and seg.center is not None and seg.radius is not None:
            radius_side = seg.radius - seg.turn * side_sign * half_w
            angle_start = seg.angle_start or 0.0
            angle_end = seg.angle_end or angle_start
            length = abs(radius_side * (angle_end - angle_start))
            remapped.append(
                TrackSegment(
                    kind="arc",
                    s_start=s_cursor,
                    s_end=s_cursor + length,
                    turn=seg.turn,
                    start=(0.0, 0.0),
                    end=(0.0, 0.0),
                    center=seg.center,
                    radius=radius_side,
                    angle_start=angle_start,
                    angle_end=angle_end,
                    left_is_inner=seg.left_is_inner,
                    length_left=length if side == "left" else seg.length_left,
                    length_right=length if side == "right" else seg.length_right,
                )
            )
            s_cursor += length

    # Recalculer start/end pour les arcs côté pour garder une géométrie cohérente
    for seg in remapped:
        if seg.kind == "arc" and seg.center is not None and seg.radius is not None:
            seg.start = (
                seg.center[0] + seg.radius * math.cos(seg.angle_start or 0.0),
                seg.center[1] + seg.radius * math.sin(seg.angle_start or 0.0),
            )
            seg.end = (
                seg.center[0] + seg.radius * math.cos(seg.angle_end or seg.angle_start or 0.0),
                seg.center[1] + seg.radius * math.sin(seg.angle_end or seg.angle_start or 0.0),
            )
    return remapped


# ---------------------------------------------------------------------------
# 5) Fonctions publiques de construction et de mesure
# ---------------------------------------------------------------------------

def build_track_from_notation(
    notation: str,
    inner_teeth: int = 144,
    outer_teeth: int = 96,
    pitch_mm_per_tooth: float = 0.65,
    steps_per_tooth: int = 3,
) -> TrackBuildResult:
    parsed = parse_track_notation(notation)
    ring = ReferenceRing(inner_teeth=float(inner_teeth), outer_teeth=float(outer_teeth), pitch_mm_per_tooth=pitch_mm_per_tooth)
    return _build_segments(parsed, ring)


def compute_track_polylines(
    track: TrackBuildResult,
    samples: int = 400,
    *,
    half_width: Optional[float] = None,
) -> Tuple[List[Point], List[Point], List[Point], float]:
    effective_half_width = half_width if half_width is not None else track.half_width
    centerline = list(_sample_segments(track.segments, samples))
    # polylignes gauche/droite via interpolation directe
    left_segments = _make_side_segments(track, "left")
    right_segments = _make_side_segments(track, "right")
    left = list(_sample_segments(left_segments, samples))
    right = list(_sample_segments(right_segments, samples))
    return centerline, left, right, effective_half_width


def compute_track_lengths(
    track: TrackBuildResult,
    inner_teeth: float,
    outer_teeth: float,
    pitch_mm_per_tooth: float,
) -> Tuple[float, float, float]:
    del inner_teeth, outer_teeth  # déjà intégrés dans le TrackBuildResult
    inner_length = track.inner_teeth * pitch_mm_per_tooth
    outer_length = track.outer_teeth * pitch_mm_per_tooth
    mid_length = 0.5 * (track.left_teeth + track.right_teeth) * pitch_mm_per_tooth
    return inner_length, mid_length, outer_length


# ---------------------------------------------------------------------------
# 6) Génération du trochoïde
# ---------------------------------------------------------------------------

def _select_side_for_relation(track: TrackBuildResult, relation: str) -> str:
    relation = relation.lower()
    if track.inner_side == "both":
        return "left" if relation == "dedans" else "right"
    if relation == "dedans":
        return track.inner_side
    return "right" if track.inner_side == "left" else "left"


def build_track_and_bundle_from_notation(
    *,
    notation: str,
    wheel_teeth: int,
    hole_index: float,
    hole_spacing_mm: float,
    steps: int,
    relation: str = "dedans",
    wheel_phase_teeth: float = 0.0,
    inner_teeth: int = 144,
    outer_teeth: int = 96,
    pitch_mm_per_tooth: float = 0.65,
) -> Tuple[TrackBuildResult, TrackRollBundle]:
    track = build_track_from_notation(
        notation,
        inner_teeth=inner_teeth,
        outer_teeth=outer_teeth,
        pitch_mm_per_tooth=pitch_mm_per_tooth,
        steps_per_tooth=3,
    )

    if steps < 2:
        steps = 2

    side = _select_side_for_relation(track, relation)
    side_segments = _make_side_segments(track, side)
    side_length = side_segments[-1].s_end if side_segments else 0.0
    if side_length <= 0:
        empty_context = TrackRollContext(
            half_width=track.half_width,
            r_wheel=0.0,
            N_wheel=max(1, int(wheel_teeth)),
            N_track=max(1, int(round(track.left_teeth if side == "left" else track.right_teeth))),
            track_length=0.0,
            sign_side=1.0 if side == "left" else -1.0,
            pitch_mm_per_tooth=pitch_mm_per_tooth,
        )
        empty = TrackRollBundle(
            stylo=[],
            centre=[],
            contact=[],
            marker0=[],
            wheel_teeth_indices=[],
            track_teeth_indices=[],
            context=empty_context,
        )
        return track, empty

    wheel_teeth = max(1, int(wheel_teeth))
    wheel_radius = (wheel_teeth * pitch_mm_per_tooth) / (2.0 * math.pi)
    track_teeth_side = track.left_teeth if side == "left" else track.right_teeth
    track_teeth_side = max(1, int(round(track_teeth_side)))
    g = math.gcd(track_teeth_side, wheel_teeth)
    laps = max(1, wheel_teeth // g)
    s_max = side_length * laps

    sign_side = 1 if side == "left" else -1
    roll_sign = -sign_side

    stylo_points: List[Point] = []
    centre_points: List[Point] = []
    contact_points: List[Point] = []
    marker0: List[Point] = []
    wheel_indices: List[int] = []
    track_indices: List[int] = []

    for i in range(steps):
        s = s_max * i / (steps - 1)
        contact, theta, normal = _interpolate_on_segments(s % side_length, side_segments)
        cx = contact[0] + sign_side * normal[0] * wheel_radius
        cy = contact[1] + sign_side * normal[1] * wheel_radius

        teeth_rolled = (s / pitch_mm_per_tooth) - float(wheel_phase_teeth) + track.origin_teeth_offset
        angle_contact = math.atan2(contact[1] - cy, contact[0] - cx)
        phi = angle_contact + roll_sign * 2.0 * math.pi * (teeth_rolled / float(wheel_teeth))
        d = max(0.0, wheel_radius - hole_index * hole_spacing_mm)

        stylo_points.append((cx + d * math.cos(phi), cy + d * math.sin(phi)))
        centre_points.append((cx, cy))
        contact_points.append(contact)
        marker0.append((cx + wheel_radius * math.cos(angle_contact), cy + wheel_radius * math.sin(angle_contact)))
        wheel_indices.append(int(math.floor((teeth_rolled % wheel_teeth + wheel_teeth) % wheel_teeth)))
        track_indices.append(int(math.floor(((s / pitch_mm_per_tooth) + track.origin_teeth_offset) % track_teeth_side)))

    context = TrackRollContext(
        half_width=track.half_width,
        r_wheel=wheel_radius,
        N_wheel=wheel_teeth,
        N_track=track_teeth_side,
        track_length=side_length,
        sign_side=sign_side,
        pitch_mm_per_tooth=pitch_mm_per_tooth,
    )

    bundle = TrackRollBundle(
        stylo=stylo_points,
        centre=centre_points,
        contact=contact_points,
        marker0=marker0,
        wheel_teeth_indices=wheel_indices,
        track_teeth_indices=track_indices,
        context=context,
    )
    return track, bundle


def generate_track_base_points(
    notation: str,
    wheel_teeth: int,
    hole_index: float,
    hole_spacing_mm: float,
    steps: int,
    relation: str = "dedans",
    output_mode: str = "stylo",
    wheel_phase_teeth: float = 0.0,
    inner_teeth: int = 144,
    outer_teeth: int = 96,
    pitch_mm_per_tooth: float = 0.65,
) -> List[Point]:
    track, bundle = build_track_and_bundle_from_notation(
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

    mode = output_mode.lower()
    if mode == "stylo":
        return bundle.stylo
    if mode == "centre":
        return bundle.centre
    if mode == "contact":
        return bundle.contact
    raise ValueError("output_mode doit être 'stylo', 'centre' ou 'contact'")
