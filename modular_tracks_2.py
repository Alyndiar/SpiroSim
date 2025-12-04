"""
modular_tracks_2.py (nouvelle version)

Génération de pistes modulaires de type SuperSpirograph + courbes associées,
basée sur un modèle géométrique par segments (arcs / barres) et non plus
sur une simple polyligne approximative.

Ce module est indépendant de Qt. Il fournit :

  - parse_track_notation(text) -> ParsedTrack
      Parse une notation de piste du type : "-18-C+D+B-...".
      La notation est volontairement simple et stable.

  - build_track_from_notation(...)
      Construit une piste géométrique à partir de la notation :
        * segments continus (arcs / barres)
        * une paramétrisation curviligne s ∈ [0, L]
        * une approximation de la piste sous forme de liste de points
          (utile pour l'aperçu dans l'éditeur Qt).

  - generate_track_base_points(...)
      Génère la courbe tracée par un stylo dans une roue qui roule
      le long de cette piste (trochoïde / cycloïde généralisée).

Hypothèses :
  - Toutes les pièces courbes (A, B, C, D, Y, ...) sont définies par
    un angle en degrés (45, 60, 90, 120, ...).
  - On suppose un pas constant par dent (pitch_mm_per_tooth).
  - Pour un anneau avec inner_teeth / outer_teeth, le nombre de dents
    pour un arc d'angle α est :
        inner_teeth * α / 360  (concave)
        outer_teeth * α / 360  (convexe)
  - L'offset en dents au début de la notation ne s'applique **qu'à la
    première pièce**, en la faisant tourner autour de son centre (arc)
    ou glisser le long de sa longueur (barre).

Limitations actuelles :
  - Les pièces Y (jonction triple), Z (terminateur) et l'opérateur '*'
    (saut de branche) sont encore parsés mais **pas** implémentés
    géométriquement. Ils sont simplement ignorés lors de la construction
    de la piste pour éviter de bloquer les notations existantes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

Point = Tuple[float, float]


@dataclass
class TrackRollContext:
    """Contexte de roulement partagé entre les générateurs de points."""

    orientation_sign: float
    sign_side: float
    half_width: float
    r_wheel: float
    N_track: int
    N_wheel: int
    track_length: float
    s_max: float
    track_offset_teeth: float
    pitch_mm_per_tooth: float


# ---------------------------------------------------------------------------
# 1) Définitions de base
# ---------------------------------------------------------------------------

@dataclass
class TrackBuildResult:
    """Résultat de la construction d'une piste modulaire."""

    # Approximation de la piste (points sur la médiane, en mm)
    points: List[Point]
    # Longueur totale de la piste (mm) sur la médiane
    total_length: float
    # Nombre "équivalent" de dents total (somme des pièces)
    total_teeth: float
    # Décalage initial (en dents) tel que parsé dans la notation
    offset_teeth: int
    # Signe de la première pièce rencontrée ("+" concave, "-" convexe)
    first_piece_sign: Optional[str] = None
    # Nouveau : segments géométriques détaillés
    segments: List["TrackSegment"] = field(default_factory=list)


@dataclass
class PieceDef:
    """
    Définition d'un type de pièce modulaire.

    arc_degrees :
      - None pour une barre droite
      - sinon, angle central (en degrés) pour les pièces courbes
        (A,B,C,D,Y, ...)

    straight_teeth :
      - nombre de dents "de rack" pour les pièces droites (E,F,Z)
      - 0.0 pour les pièces courbes
    """

    name: str
    arc_degrees: Optional[float] = None
    straight_teeth: float = 0.0
    special_type: str = "normal"   # "normal", "Y", "Z"


# Angles approximatifs pour les pièces courbes
# E/F/Z : barres (ou fin de piste) définies par un nombre de dents "de rack"
PIECES = {
    "A": PieceDef("A", arc_degrees=45.0),
    "B": PieceDef("B", arc_degrees=60.0),
    "C": PieceDef("C", arc_degrees=90.0),
    "D": PieceDef("D", arc_degrees=120.0),
    "E": PieceDef("E", arc_degrees=None, straight_teeth=20.0),
    "F": PieceDef("F", arc_degrees=None, straight_teeth=56.0),
    # Y : 3 branches concaves de 60°, opposées de 120° chacune.
    "Y": PieceDef("Y", arc_degrees=60.0, special_type="Y"),
    # Z : "end piece" – demi-cercle, 14 dents de rack
    "Z": PieceDef("Z", arc_degrees=None, straight_teeth=14.0, special_type="Z"),
}


@dataclass
class ParsedElement:
    kind: str                    # "piece" ou "branch"
    sign: Optional[str] = None   # "+" ou "-"
    piece_name: Optional[str] = None


@dataclass
class ParsedTrack:
    offset_teeth: int
    elements: List[ParsedElement]


# Un segment de piste (médiane) – soit un arc de cercle, soit un segment droit.
@dataclass
class TrackSegment:
    kind: str                   # "arc" ou "line"
    s_start: float              # longueur curviligne de début
    s_end: float                # longueur curviligne de fin
    sign: Optional[str] = None  # signe de la pièce "+" ou "-" (pour les longueurs)

    # Pour les arcs :
    O: Optional[Point] = None   # centre du cercle médian
    rM: float = 0.0             # rayon médian
    phi_start: float = 0.0
    phi_end: float = 0.0
    sigma_curve: int = 0        # +1 (tourne à gauche), -1 (tourne à droite)

    # Pour les segments droits :
    P0: Optional[Point] = None
    P1: Optional[Point] = None

    # Informations de bord de contact pour la roue :
    side: str = "concave"       # "concave", "convexe" ou "bar"
    R_track: float = 0.0        # rayon effectif du bord (mm)
    sigma_roll: int = 1         # signe pour la phase de rotation (pas encore utilisé)


# ---------------------------------------------------------------------------
# 2) Parsing de la notation
# ---------------------------------------------------------------------------

def parse_track_notation(text: str) -> ParsedTrack:
    """
    Parse une notation du type : -18-C+D+B-C+D+...

    offset_teeth :
      - entier signé (peut être 0 ou absent)
      - il décrit de combien de dents il faut faire tourner la
        **première pièce** autour de son centre (arc) ou déplacer
        une barre le long de sa longueur.
    éléments :
      - "+X" ou "-X" pour les pièces (X = A, B, C, D, E, F, Y, Z)
      - "*" pour sauter de branche sur les pièces spéciales (pour
        l'instant, simplement enregistré comme "branch" et ignoré
        côté géométrie).
    """
    s = text.strip().replace(" ", "").upper()
    if not s:
        return ParsedTrack(0, [])

    idx = 0
    n = len(s)

    # 1) décalage initial (entier signé, optionnel)
    offset_teeth = 0
    sign = 1
    start_idx = idx

    if idx < n and s[idx] in "+-":
        sign = -1 if s[idx] == "-" else 1
        idx += 1
        start_idx = idx

    while idx < n and s[idx].isdigit():
        idx += 1

    if idx > start_idx:
        try:
            offset_teeth = sign * int(s[start_idx:idx])
        except ValueError:
            offset_teeth = 0
    else:
        # pas de décalage explicite, on repart du début
        idx = 0

    elements: List[ParsedElement] = []

    # 2) séquence de pièces et de sauts de branche
    while idx < n:
        ch = s[idx]
        if ch == "*":
            elements.append(ParsedElement(kind="branch"))
            idx += 1
            continue

        if ch in "+-":
            sign_char = ch
            idx += 1
            if idx >= n:
                break
            piece_name = s[idx]
            idx += 1
            if piece_name not in PIECES:
                # symbole non reconnu -> ignoré
                continue
            elements.append(
                ParsedElement(kind="piece", sign=sign_char, piece_name=piece_name)
            )
        else:
            # caractère non reconnu : on l'ignore
            idx += 1

    return ParsedTrack(offset_teeth=offset_teeth, elements=elements)


# ---------------------------------------------------------------------------
# 3) Construction géométrique de la piste
# ---------------------------------------------------------------------------

@dataclass
class _PoseState:
    """État local lors de la pose des pièces sur la médiane de la piste."""

    P: Point         # point courant sur la médiane
    T: Point         # tangente unité (direction de la progression)
    N: Point         # normale unité (à gauche de T)
    s: float         # longueur curviligne déjà parcourue (mm)


def _normalize(vx: float, vy: float) -> Point:
    n = math.hypot(vx, vy)
    if n == 0:
        return 0.0, 0.0
    return vx / n, vy / n


def _track_orientation_sign(track: TrackBuildResult) -> float:
    """Retourne +1 pour une piste CCW, -1 pour CW (0 => +1 par défaut)."""

    def _turn_sum_rad(segments: List[TrackSegment]) -> float:
        turn = 0.0
        for seg in segments:
            if seg.kind == "arc":
                turn += seg.phi_end - seg.phi_start
        return turn

    turn_rad = _turn_sum_rad(track.segments)
    turn_deg = math.degrees(turn_rad)
    if abs(turn_deg) >= 300.0:
        return 1.0 if turn_deg > 0.0 else -1.0

    pts = track.points or []
    if len(pts) < 3:
        return 1.0
    area = 0.0
    for i, (x0, y0) in enumerate(pts):
        x1, y1 = pts[(i + 1) % len(pts)]
        area += x0 * y1 - x1 * y0
    return 1.0 if area >= 0.0 else -1.0


def _build_segments_from_parsed(
    parsed: ParsedTrack,
    inner_teeth: float,
    outer_teeth: float,
    pitch_mm_per_tooth: float,
) -> TrackBuildResult:
    """
    Construit une liste de TrackSegment à partir d'une ParsedTrack.

    Hypothèses :
      - on parcourt la piste dans le sens "Spirograph" :
          * au point s=0, la tangente est orientée vers la droite (+X),
            la normale vers le haut (+Y).
      - les pièces Y / Z / '*' ne sont pas encore gérées géométriquement.
    """
    segments: List[TrackSegment] = []
    total_teeth = 0.0

    # Rayons concave / convexe et valeurs dérivées
    r_in = (inner_teeth * pitch_mm_per_tooth) / (2.0 * math.pi)
    r_out = (outer_teeth * pitch_mm_per_tooth) / (2.0 * math.pi)
    dR = r_out - r_in
    rM = (r_out + r_in) * 0.5

    # Pose initiale : on part de (0,0), tangente vers la droite, normale vers le haut.
    pose = _PoseState(P=(0.0, 0.0), T=(1.0, 0.0), N=(0.0, 1.0), s=0.0)
    first_piece_sign: Optional[str] = None

    # offset pour la première pièce uniquement
    remaining_offset = parsed.offset_teeth
    first_piece_done = False

    for elem in parsed.elements:
        if elem.kind == "branch":
            # Les branches (*) ne sont pas encore gérées : on les ignore simplement.
            continue

        if elem.kind != "piece" or elem.piece_name is None or elem.sign is None:
            continue

        name = elem.piece_name
        pdef = PIECES.get(name)
        if pdef is None:
            continue

        if pdef.special_type in ("Y", "Z"):
            # Pièces spéciales non implémentées pour l'instant : on ignore.
            continue

        sign_char = elem.sign  # "+" (concave) ou "-" (convexe)

        # ----------------------------------------
        # Cas : pièce courbe (A,B,C,D)
        # ----------------------------------------
        if pdef.arc_degrees is not None:
            angle_deg = pdef.arc_degrees
            angle_rad = math.radians(angle_deg)

            # Bord utilisé pour le contact : concave (+) -> inner, convexe (-) -> outer
            if sign_char == "+":
                side = "concave"
                R_track = r_in
                if first_piece_sign is None:
                    first_piece_sign = "+"
                # On choisit : concave = piste tourne "vers la droite" globalement,
                # ce qui correspond à sigma_curve = -1 (arc horaire).
                sigma_curve = -1
            else:
                side = "convexe"
                R_track = r_out
                if first_piece_sign is None:
                    first_piece_sign = "-"
                # convexe = piste tourne "vers la gauche", sigma_curve = +1 (anti-horaire)
                sigma_curve = +1

            T_x, T_y = pose.T
            N_x, N_y = pose.N
            P_x, P_y = pose.P

            # Centre de l'arc médian
            O_x = P_x + sigma_curve * rM * N_x
            O_y = P_y + sigma_curve * rM * N_y
            O = (O_x, O_y)

            # rayon radial initial
            R0_x = P_x - O_x
            R0_y = P_y - O_y
            phi_start = math.atan2(R0_y, R0_x)

            # --- offset sur la première pièce (rotation autour du centre) ---
            if not first_piece_done and remaining_offset != 0:
                # longueur le long du bord de contact
                l_off = remaining_offset * pitch_mm_per_tooth
                # l_off = R_track * delta_phi
                delta_phi = l_off / max(R_track, 1e-9)
                # Convention : concave (+) -> offset anti-horaire, convexe (-) -> horaire
                if side == "concave":
                    phi_start += delta_phi
                else:  # convexe
                    phi_start -= delta_phi
                remaining_offset = 0  # consommé

            phi_end = phi_start + sigma_curve * angle_rad

            # Longueur sur le bord de contact (mm)
            seg_length = abs(R_track * angle_rad)
            s_start = pose.s
            s_end = s_start + seg_length

            seg = TrackSegment(
                kind="arc",
                s_start=s_start,
                s_end=s_end,
                sign=sign_char,
                O=O,
                rM=rM,
                phi_start=phi_start,
                phi_end=phi_end,
                sigma_curve=sigma_curve,
                P0=(P_x, P_y),
                side=side,
                R_track=R_track,
                sigma_roll=1,
            )
            segments.append(seg)
            total_teeth += (inner_teeth if side == "concave" else outer_teeth) * (angle_deg / 360.0)

            # Mettre à jour la pose à la fin de l'arc
            phi = phi_end
            # Nouveau point sur la médiane
            P_x = O_x + rM * math.cos(phi)
            P_y = O_y + rM * math.sin(phi)
            P = (P_x, P_y)

            # Tangente : dépend du sens de parcours
            if sigma_curve == +1:
                # anti-horaire : T = (-sin φ, cos φ)
                T_x, T_y = -math.sin(phi), math.cos(phi)
            else:
                # horaire : T = (sin φ, -cos φ)
                T_x, T_y = math.sin(phi), -math.cos(phi)
            T_x, T_y = _normalize(T_x, T_y)
            # Normale à gauche de T
            N_x, N_y = -T_y, T_x

            pose = _PoseState(P=P, T=(T_x, T_y), N=(N_x, N_y), s=s_end)
            first_piece_done = True

        # ----------------------------------------
        # Cas : barre droite (E,F, ou toute pièce avec straight_teeth>0)
        # ----------------------------------------
        elif pdef.straight_teeth > 0.0:
            side = "bar"
            R_track = dR * 0.5  # on peut assimiler une barre à un "rack" de largeur dR

            if first_piece_sign is None:
                first_piece_sign = sign_char

            # longueur de la barre (mm)
            seg_length = pdef.straight_teeth * pitch_mm_per_tooth

            T_x, T_y = pose.T
            N_x, N_y = pose.N
            P_x, P_y = pose.P
            P0 = (P_x, P_y)

            # --- offset sur la première pièce (translation le long de la barre) ---
            if not first_piece_done and remaining_offset != 0:
                delta = remaining_offset * pitch_mm_per_tooth
                # Un offset positif déplace le point de départ dans la direction +T
                P_x += delta * T_x
                P_y += delta * T_y
                P0 = (P_x, P_y)
                remaining_offset = 0

            P1_x = P_x + seg_length * T_x
            P1_y = P_y + seg_length * T_y
            P1 = (P1_x, P1_y)

            s_start = pose.s
            s_end = s_start + seg_length

            seg = TrackSegment(
                kind="line",
                s_start=s_start,
                s_end=s_end,
                sign=sign_char,
                P0=P0,
                P1=P1,
                side=side,
                R_track=R_track,
                sigma_roll=1,
            )
            segments.append(seg)
            total_teeth += pdef.straight_teeth

            pose = _PoseState(P=P1, T=(T_x, T_y), N=(N_x, N_y), s=s_end)
            first_piece_done = True

        else:
            # pièce sans définition utilisable
            continue

    if not segments:
        # Piste vide -> renvoyer un résultat trivial
        return TrackBuildResult(
            points=[],
            total_length=0.0,
            total_teeth=0.0,
            offset_teeth=parsed.offset_teeth,
            first_piece_sign=first_piece_sign,
            segments=[],
        )

    total_length = segments[-1].s_end

    # ------------------------------------------------------------------
    # Génération d'une approximation "points" de la médiane de la piste
    # (utile pour l'aperçu dans l'éditeur Qt)
    # ------------------------------------------------------------------
    num_samples = max(200, int(total_length / (pitch_mm_per_tooth * 0.5)))
    pts: List[Point] = []
    for i in range(num_samples + 1):
        s = (total_length * i) / float(num_samples)
        C, _, _ = _interpolate_on_segments(s, segments)
        pts.append(C)

    # Recentrer la piste : barycentre -> (0,0)
    # Calcul barycentre
    bx = sum(x for (x, _) in pts) / len(pts)
    by = sum(y for (_, y) in pts) / len(pts)
    pts_centered: List[Point] = []
    for (x, y) in pts:
        pts_centered.append((x - bx, y - by))

    # Appliquer la même translation aux segments
    segments_centered: List[TrackSegment] = []
    for seg in segments:
        s_start = seg.s_start
        s_end = seg.s_end
        if seg.kind == "arc" and seg.O is not None:
            O_x, O_y = seg.O
            # recentrage
            O_x -= bx
            O_y -= by
            new_seg = TrackSegment(
                kind="arc",
                s_start=s_start,
                s_end=s_end,
                sign=seg.sign,
                O=(O_x, O_y),
                rM=seg.rM,
                phi_start=seg.phi_start,
                phi_end=seg.phi_end,
                sigma_curve=seg.sigma_curve,
                side=seg.side,
                R_track=seg.R_track,
                sigma_roll=seg.sigma_roll,
            )
        elif seg.kind == "line" and seg.P0 is not None and seg.P1 is not None:
            x0, y0 = seg.P0
            x1, y1 = seg.P1
            # recentrage
            x0 -= bx
            y0 -= by
            x1 -= bx
            y1 -= by
            new_seg = TrackSegment(
                kind="line",
                s_start=s_start,
                s_end=s_end,
                sign=seg.sign,
                P0=(x0, y0),
                P1=(x1, y1),
                side=seg.side,
                R_track=seg.R_track,
                sigma_roll=seg.sigma_roll,
            )
        else:
            new_seg = seg
        segments_centered.append(new_seg)

    return TrackBuildResult(
        points=pts_centered,
        total_length=total_length,
        total_teeth=total_teeth,
        offset_teeth=parsed.offset_teeth,
        first_piece_sign=first_piece_sign,
        segments=segments_centered,
    )


def _build_roll_context(
    track: TrackBuildResult,
    *,
    relation: str,
    wheel_teeth: int,
    inner_teeth: int,
    outer_teeth: int,
    pitch_mm_per_tooth: float,
    track_length_override: Optional[float] = None,
    track_teeth_override: Optional[float] = None,
) -> TrackRollContext:
    """Prépare les paramètres partagés pour le roulement sans glissement."""

    relation = relation.lower()

    r_in = (inner_teeth * pitch_mm_per_tooth) / (2.0 * math.pi)
    r_out = (outer_teeth * pitch_mm_per_tooth) / (2.0 * math.pi)
    dR = r_out - r_in
    half_width = abs(dR) * 0.5 if abs(dR) > 0.0 else max(pitch_mm_per_tooth, 1.0)

    r_wheel = (wheel_teeth * pitch_mm_per_tooth) / (2.0 * math.pi)

    track_teeth_value = (
        track_teeth_override if track_teeth_override is not None else track.total_teeth
    )

    if track_teeth_value > 0:
        N_track = max(1, int(round(track_teeth_value)))
    else:
        N_track = max(1, int(wheel_teeth))

    N_wheel = max(1, int(wheel_teeth))
    g = math.gcd(N_track, N_wheel)
    if g <= 0:
        g = 1
    nb_laps = N_wheel // g if N_wheel >= g else 1
    if nb_laps < 1:
        nb_laps = 1

    orientation_sign = _track_orientation_sign(track)
    sign_side = orientation_sign if relation == "dedans" else -orientation_sign

    L = track_length_override if track_length_override is not None else track.total_length
    s_max = L * float(nb_laps)

    return TrackRollContext(
        orientation_sign=orientation_sign,
        sign_side=sign_side,
        half_width=half_width,
        r_wheel=r_wheel,
        N_track=N_track,
        N_wheel=N_wheel,
        track_length=L,
        s_max=s_max,
        track_offset_teeth=float(track.offset_teeth or 0.0),
        pitch_mm_per_tooth=pitch_mm_per_tooth,
    )


def build_track_from_notation(
    notation: str,
    inner_teeth: int = 96,
    outer_teeth: int = 144,
    pitch_mm_per_tooth: float = 0.65,
    steps_per_tooth: int = 3,  # non utilisé dans cette version, conservé pour compatibilité
) -> TrackBuildResult:
    """
    Helper direct : parse puis construit la piste géométrique.

    La valeur de retour contient :
      - une approximation "points" de la médiane de la piste (champ .points)
      - la longueur totale (mm)
      - le nombre de dents "équivalent" (inner/outer) pour la piste
      - la liste des segments géométriques (champ .segments)
    """
    parsed = parse_track_notation(notation)
    return _build_segments_from_parsed(parsed, inner_teeth, outer_teeth, pitch_mm_per_tooth)


# ---------------------------------------------------------------------------
# 4) Mesures de piste (longueurs par offset)
# ---------------------------------------------------------------------------

def compute_track_lengths(
    track: TrackBuildResult,
    inner_teeth: float,
    outer_teeth: float,
    pitch_mm_per_tooth: float,
) -> Tuple[float, float, float]:
    """Retourne les longueurs des pistes intérieure, médiane et extérieure.

    Les longueurs sont calculées en additionnant le nombre de dents de chaque
    segment selon le signe de la pièce :

      * signe "+" : la piste intérieure suit le côté concave, l'extérieure le
        côté convexe ;
      * signe "-" : la piste intérieure suit le côté convexe, l'extérieure le
        côté concave.

    La piste intérieure finale est le chemin totalisant le **plus petit nombre
    de dents** ; la piste extérieure est le chemin totalisant le plus grand
    nombre de dents. La piste médiane est la moyenne arithmétique des deux.
    """

    if pitch_mm_per_tooth <= 0:
        return 0.0, 0.0, 0.0

    track_inner_teeth = 0.0
    track_outer_teeth = 0.0

    for seg in track.segments or []:
        if seg.kind == "arc" and seg.phi_end is not None:
            angle_rad = abs(seg.phi_end - seg.phi_start)
            angle_deg = math.degrees(angle_rad)
            concave_teeth = inner_teeth * (angle_deg / 360.0)
            convex_teeth = outer_teeth * (angle_deg / 360.0)
            sign = seg.sign or (
                "+" if seg.side == "concave" else "-" if seg.side == "convexe" else None
            )
            if sign == "-":
                track_inner_teeth += convex_teeth
                track_outer_teeth += concave_teeth
            else:
                track_inner_teeth += concave_teeth
                track_outer_teeth += convex_teeth
        elif seg.kind == "line":
            length = max(seg.s_end - seg.s_start, 0.0)
            teeth = length / pitch_mm_per_tooth
            track_inner_teeth += teeth
            track_outer_teeth += teeth

    # Le plus petit total est considéré comme la piste intérieure
    if track_inner_teeth > track_outer_teeth:
        track_inner_teeth, track_outer_teeth = track_outer_teeth, track_inner_teeth

    mid_teeth = 0.5 * (track_inner_teeth + track_outer_teeth)

    return (
        track_inner_teeth * pitch_mm_per_tooth,
        mid_teeth * pitch_mm_per_tooth,
        track_outer_teeth * pitch_mm_per_tooth,
    )


def _estimate_track_half_width(segments: List["TrackSegment"]) -> float:
    """Estime une demi-largeur de piste à partir de segments connus."""

    for seg in segments:
        if seg.kind == "arc":
            width = abs(seg.rM - seg.R_track) * 2.0
        elif seg.kind == "line":
            width = abs(seg.R_track) * 2.0
        else:
            continue
        if width > 0:
            return width * 0.5
    return 5.0  # valeur de repli (mm)


def compute_track_polylines(
    track: TrackBuildResult,
    samples: int = 400,
    *,
    half_width: Optional[float] = None,
) -> Tuple[List[Point], List[Point], List[Point], float]:
    """Retourne (centre, côté intérieur, côté extérieur, demi-largeur).

    La demi-largeur est prise depuis l'argument fourni ou estimée depuis les
    segments. Les polylignes sont dérivées directement des segments pour
    rester cohérentes avec la géométrie utilisée par la génération du trochoïde.
    """

    segments = track.segments
    effective_half_width = half_width if (half_width and half_width > 0.0) else None
    if effective_half_width is None:
        effective_half_width = _estimate_track_half_width(segments)

    L = track.total_length
    centerline: List[Point] = track.points if track.points else []

    if not centerline:
        for i in range(samples + 1):
            s = (L * i) / float(max(samples, 1))
            C, _, _ = _interpolate_on_segments(s, segments)
            centerline.append(C)

    inner: List[Point] = []
    outer: List[Point] = []

    for i in range(samples + 1):
        s = (L * i) / float(max(samples, 1))
        C, _, N = _interpolate_on_segments(s, segments)
        x, y = C
        nx, ny = N
        inner.append((x - nx * effective_half_width, y - ny * effective_half_width))
        outer.append((x + nx * effective_half_width, y + ny * effective_half_width))

    return centerline, inner, outer, effective_half_width


# ---------------------------------------------------------------------------
# 4.bis) Re-paramétrisation des segments selon le côté utilisé
# ---------------------------------------------------------------------------

def _parameterize_segments_for_relation(
    track: TrackBuildResult,
    relation: str,
    inner_teeth: float,
    outer_teeth: float,
    pitch_mm_per_tooth: float,
) -> Tuple[List[TrackSegment], float, float]:
    """Crée une copie des segments avec s recalculé pour le côté choisi.

    Le paramètre ``relation`` indique si la roue roule *dedans* (piste
    intérieure) ou *dehors* (piste extérieure). Les longueurs curvilignes
    (s_start/s_end) sont reconstruites en fonction du rayon de contact
    correspondant, et le total de dents est recalculé en cohérence.
    """

    if not track.segments or pitch_mm_per_tooth <= 0:
        return list(track.segments or []), track.total_length, track.total_teeth

    use_inner = relation.strip().lower() == "dedans"
    r_in = (inner_teeth * pitch_mm_per_tooth) / (2.0 * math.pi)
    r_out = (outer_teeth * pitch_mm_per_tooth) / (2.0 * math.pi)

    s_cursor = 0.0
    total_teeth = 0.0
    remapped: List[TrackSegment] = []

    for seg in track.segments:
        seg_length = 0.0
        seg_teeth = 0.0
        sign = seg.sign or (
            "+" if seg.side == "concave" else "-" if seg.side == "convexe" else None
        )

        if seg.kind == "arc" and seg.phi_end is not None:
            angle_rad = abs(seg.phi_end - seg.phi_start)
            if sign == "+":
                radius = r_in if use_inner else r_out
                seg_teeth = (inner_teeth if use_inner else outer_teeth) * (
                    math.degrees(angle_rad) / 360.0
                )
            elif sign == "-":
                radius = r_out if use_inner else r_in
                seg_teeth = (outer_teeth if use_inner else inner_teeth) * (
                    math.degrees(angle_rad) / 360.0
                )
            else:
                radius = seg.R_track if seg.R_track > 0 else (r_in + r_out) * 0.5
                seg_teeth = abs(radius * angle_rad) / pitch_mm_per_tooth

            seg_length = abs(radius * angle_rad)

            new_seg = TrackSegment(
                kind="arc",
                s_start=s_cursor,
                s_end=s_cursor + seg_length,
                sign=seg.sign,
                O=seg.O,
                rM=seg.rM,
                phi_start=seg.phi_start,
                phi_end=seg.phi_end,
                sigma_curve=seg.sigma_curve,
                P0=seg.P0,
                side=seg.side,
                R_track=radius,
                sigma_roll=seg.sigma_roll,
            )
        elif seg.kind == "line" and seg.P0 is not None and seg.P1 is not None:
            x0, y0 = seg.P0
            x1, y1 = seg.P1
            seg_length = math.hypot(x1 - x0, y1 - y0)
            seg_teeth = seg_length / pitch_mm_per_tooth
            new_seg = TrackSegment(
                kind="line",
                s_start=s_cursor,
                s_end=s_cursor + seg_length,
                sign=seg.sign,
                P0=seg.P0,
                P1=seg.P1,
                side=seg.side,
                R_track=seg.R_track,
                sigma_roll=seg.sigma_roll,
            )
        else:
            new_seg = seg

        remapped.append(new_seg)
        s_cursor += seg_length
        total_teeth += seg_teeth

    return remapped, s_cursor, total_teeth


# ---------------------------------------------------------------------------
# 5) Interpolation sur la piste et génération de trochoïdes
# ---------------------------------------------------------------------------

def _interpolate_on_segments(
    s: float,
    segments: List[TrackSegment],
) -> Tuple[Point, float, Point]:
    """
    Pour une valeur de s (0 <= s <= s_total), renvoie :

      - C(s) : point sur la médiane
      - theta : angle de la tangente en ce point (angle math standard,
                0 = +X, sens anti-horaire)
      - N : normale unité vers la gauche de la tangente

    s en dehors de [0, s_total] est clampé.
    """
    if not segments:
        return (0.0, 0.0), 0.0, (0.0, 1.0)

    s_total = segments[-1].s_end
    if s <= 0.0:
        seg = segments[0]
        s_local = 0.0
    elif s >= s_total:
        seg = segments[-1]
        s_local = seg.s_end - seg.s_start
    else:
        seg = segments[0]
        for sg in segments:
            if sg.s_start <= s <= sg.s_end:
                seg = sg
                break
        s_local = s - seg.s_start
        if s_local < 0.0:
            s_local = 0.0

    if seg.kind == "line" and seg.P0 is not None and seg.P1 is not None:
        x0, y0 = seg.P0
        x1, y1 = seg.P1
        L = seg.s_end - seg.s_start
        if L <= 0:
            return (x0, y0), 0.0, (0.0, 1.0)
        t = s_local / L
        x = x0 + (x1 - x0) * t
        y = y0 + (y1 - y0) * t
        tx, ty = _normalize(x1 - x0, y1 - y0)
        theta = math.atan2(ty, tx)
        nx, ny = -ty, tx
        return (x, y), theta, (nx, ny)

    if seg.kind == "arc" and seg.O is not None:
        O_x, O_y = seg.O
        L = seg.s_end - seg.s_start
        if L <= 0:
            # point unique au début de l'arc
            phi = seg.phi_start
        else:
            t = s_local / L
            phi = seg.phi_start + (seg.phi_end - seg.phi_start) * t

        rM = seg.rM
        x = O_x + rM * math.cos(phi)
        y = O_y + rM * math.sin(phi)

        # Tangente selon le sens de parcours
        if seg.sigma_curve == +1:
            # anti-horaire : T = (-sin φ, cos φ)
            tx, ty = -math.sin(phi), math.cos(phi)
        else:
            # horaire : T = (sin φ, -cos φ)
            tx, ty = math.sin(phi), -math.cos(phi)
        tx, ty = _normalize(tx, ty)
        theta = math.atan2(ty, tx)
        nx, ny = -ty, tx
        return (x, y), theta, (nx, ny)

    # Fallback
    return (0.0, 0.0), 0.0, (0.0, 1.0)


@dataclass
class TrackRollBundle:
    stylo: List[Point]
    centre: List[Point]
    contact: List[Point]
    marker0: List[Point]
    wheel_teeth_indices: List[int]
    track_teeth_indices: List[int]
    context: TrackRollContext


def _generate_track_roll_bundle(
    *,
    track: TrackBuildResult,
    notation: str,
    wheel_teeth: int,
    hole_index: float,
    hole_spacing_mm: float,
    steps: int,
    relation: str = "dedans",
    wheel_phase_teeth: float = 0.0,
    inner_teeth: int = 96,
    outer_teeth: int = 144,
    pitch_mm_per_tooth: float = 0.65,
) -> TrackRollBundle:
    """Calcule tous les points nécessaires à l'animation (stylo, centre...)."""

    if steps <= 1:
        steps = 2

    segments_for_relation, track_length_rel, track_teeth_rel = _parameterize_segments_for_relation(
        track,
        relation,
        inner_teeth,
        outer_teeth,
        pitch_mm_per_tooth,
    )

    context = _build_roll_context(
        track,
        relation=relation,
        wheel_teeth=wheel_teeth,
        inner_teeth=inner_teeth,
        outer_teeth=outer_teeth,
        pitch_mm_per_tooth=pitch_mm_per_tooth,
        track_length_override=track_length_rel,
        track_teeth_override=track_teeth_rel,
    )

    if not segments_for_relation or context.track_length <= 0:
        return TrackRollBundle(
            stylo=[],
            centre=[],
            contact=[],
            marker0=[],
            wheel_teeth_indices=[],
            track_teeth_indices=[],
            context=context,
        )

    r_wheel = context.r_wheel
    d = max(0.0, r_wheel - hole_index * hole_spacing_mm)

    N_w = context.N_wheel
    N_track = context.N_track
    roll_sign = -context.sign_side
    L = context.track_length
    s_max = context.s_max
    track_offset_teeth = context.track_offset_teeth
    pitch = context.pitch_mm_per_tooth

    stylo_points: List[Point] = []
    wheel_centers: List[Point] = []
    contacts: List[Point] = []
    marker0: List[Point] = []
    wheel_teeth_indices: List[int] = []
    track_teeth_indices: List[int] = []

    for i in range(steps):
        s = s_max * i / (steps - 1)
        C, theta, N_vec = _interpolate_on_segments(s % L, segments_for_relation)
        x_track, y_track = C
        nx, ny = N_vec

        contact_x = x_track + context.sign_side * nx * context.half_width
        contact_y = y_track + context.sign_side * ny * context.half_width

        cx = contact_x + context.sign_side * nx * r_wheel
        cy = contact_y + context.sign_side * ny * r_wheel

        wheel_centers.append((cx, cy))
        contacts.append((contact_x, contact_y))

        teeth_rolled = (s / pitch) - float(wheel_phase_teeth) + track_offset_teeth
        angle_contact = math.atan2(contact_y - cy, contact_x - cx)
        phi = angle_contact + roll_sign * 2.0 * math.pi * (teeth_rolled / float(N_w))

        stylo_points.append((cx + d * math.cos(phi), cy + d * math.sin(phi)))
        marker0.append((cx + r_wheel * math.cos(angle_contact), cy + r_wheel * math.sin(angle_contact)))

        wheel_teeth_indices.append(int(math.floor((teeth_rolled % N_w + N_w) % N_w)))
        track_teeth_indices.append(int(math.floor(((s / pitch) + track_offset_teeth) % N_track)))

    return TrackRollBundle(
        stylo=stylo_points,
        centre=wheel_centers,
        contact=contacts,
        marker0=marker0,
        wheel_teeth_indices=wheel_teeth_indices,
        track_teeth_indices=track_teeth_indices,
        context=context,
    )


def build_track_and_bundle_from_notation(
    *,
    notation: str,
    wheel_teeth: int,
    hole_index: float,
    hole_spacing_mm: float,
    steps: int,
    relation: str = "dedans",
    wheel_phase_teeth: float = 0.0,
    inner_teeth: int = 96,
    outer_teeth: int = 144,
    pitch_mm_per_tooth: float = 0.65,
) -> Tuple[TrackBuildResult, TrackRollBundle]:
    """Construit la piste et le bundle de roulage associé en une seule étape."""

    track = build_track_from_notation(
        notation,
        inner_teeth=inner_teeth,
        outer_teeth=outer_teeth,
        pitch_mm_per_tooth=pitch_mm_per_tooth,
    )

    bundle = _generate_track_roll_bundle(
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
    inner_teeth: int = 96,
    outer_teeth: int = 144,
    pitch_mm_per_tooth: float = 0.65,
) -> List[Point]:
    """
    Génère la courbe trochoïdale le long d'une piste modulaire.

    notation :
      - chaîne décrivant la piste (ex: "-18-C+D+B-C+D+...").

    wheel_teeth :
      - nombre de dents de la roue mobile (celle qui porte les trous).

    hole_index :
      - index (float, peut être négatif) du trou utilisé sur la roue.
        hole_index = 0 => trou au niveau de la pointe de la dent,
        chaque +1 = +hole_spacing_mm vers le centre.

    relation :
      - "dedans" : roue à l'intérieur de la piste
      - "dehors" : roue à l'extérieur
    """
    _, bundle = build_track_and_bundle_from_notation(
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
    raise ValueError("output_mode doit être 'stylo', 'contact' ou 'centre'")
