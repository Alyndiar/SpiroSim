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
    géométriquement : un NotImplementedError sera levé si on les
    rencontre lors de la construction de la piste.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

Point = Tuple[float, float]


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
                # On choisit : concave = piste tourne "vers la droite" globalement,
                # ce qui correspond à sigma_curve = -1 (arc horaire).
                sigma_curve = -1
            else:
                side = "convexe"
                R_track = r_out
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
        return TrackBuildResult(points=[], total_length=0.0, total_teeth=0.0, offset_teeth=parsed.offset_teeth, segments=[])

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

    # Recentrer et orienter la piste :
    #  - barycentre -> (0,0)
    #  - puis rotation globale fixe de +pi/2 pour avoir l'angle 0 vers +Y
    #    (convention Spirograph : point 0 "en haut".)
    # Calcul barycentre
    bx = sum(x for (x, _) in pts) / len(pts)
    by = sum(y for (_, y) in pts) / len(pts)
    pts_centered: List[Point] = []
    for (x, y) in pts:
        pts_centered.append((x - bx, y - by))

    # Rotation globale fixée : angle 0 vers +Y (pi/2 par rapport à +X)
    rot = math.pi / 2.0
    cos_r = math.cos(rot)
    sin_r = math.sin(rot)

    rotated: List[Point] = []
    for (x, y) in pts_centered:
        xr = x * cos_r - y * sin_r
        yr = x * sin_r + y * cos_r
        rotated.append((xr, yr))

    # Appliquer la même translation+rotation aux segments
    segments_rot: List[TrackSegment] = []
    for seg in segments:
        s_start = seg.s_start
        s_end = seg.s_end
        if seg.kind == "arc" and seg.O is not None:
            O_x, O_y = seg.O
            # recentrage
            O_x -= bx
            O_y -= by
            # rotation
            Or_x = O_x * cos_r - O_y * sin_r
            Or_y = O_x * sin_r + O_y * cos_r
            # angles : on ajoute la rotation globale
            phi_start = seg.phi_start + rot
            phi_end = seg.phi_end + rot
            new_seg = TrackSegment(
                kind="arc",
                s_start=s_start,
                s_end=s_end,
                O=(Or_x, Or_y),
                rM=seg.rM,
                phi_start=phi_start,
                phi_end=phi_end,
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
            # rotation
            xr0 = x0 * cos_r - y0 * sin_r
            yr0 = x0 * sin_r + y0 * cos_r
            xr1 = x1 * cos_r - y1 * sin_r
            yr1 = x1 * sin_r + y1 * cos_r
            new_seg = TrackSegment(
                kind="line",
                s_start=s_start,
                s_end=s_end,
                P0=(xr0, yr0),
                P1=(xr1, yr1),
                side=seg.side,
                R_track=seg.R_track,
                sigma_roll=seg.sigma_roll,
            )
        else:
            new_seg = seg
        segments_rot.append(new_seg)

    return TrackBuildResult(
        points=rotated,
        total_length=total_length,
        total_teeth=total_teeth,
        offset_teeth=parsed.offset_teeth,
        segments=segments_rot,
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
# 4) Interpolation sur la piste et génération de trochoïdes
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
    if steps <= 1:
        steps = 2

    # Construire la piste géométrique
    track = build_track_from_notation(
        notation,
        inner_teeth=inner_teeth,
        outer_teeth=outer_teeth,
        pitch_mm_per_tooth=pitch_mm_per_tooth,
    )
    segments = track.segments
    if not segments:
        return []

    # Rayons de l'anneau de référence
    r_in = (inner_teeth * pitch_mm_per_tooth) / (2.0 * math.pi)
    r_out = (outer_teeth * pitch_mm_per_tooth) / (2.0 * math.pi)
    dR = r_out - r_in

    # Rayon de la roue mobile
    r_wheel = (wheel_teeth * pitch_mm_per_tooth) / (2.0 * math.pi)

    # Distance du stylo au centre de la roue :
    #  - piste intérieure/extérieur -> on utilise le rayon de la roue comme base
    R_tip = r_wheel
    d = R_tip - hole_index * hole_spacing_mm
    if d < 0.0:
        d = 0.0

    # Longueur totale de la piste (sur la médiane)
    L = track.total_length
    if L <= 0:
        return []

    # Nombre "équivalent" de dents pour la piste
    if track.total_teeth > 0:
        N_track = max(1, int(round(track.total_teeth)))
    else:
        N_track = wheel_teeth

    N_w = max(1, int(wheel_teeth))
    g = math.gcd(N_track, N_w)
    if g <= 0:
        g = 1
    nb_laps = N_w // g if N_w >= g else 1
    if nb_laps < 1:
        nb_laps = 1

    s_max = L * float(nb_laps)

    base_points: List[Point] = []

    mode = output_mode.lower()
    if mode not in {"stylo", "contact", "centre"}:
        raise ValueError("output_mode doit être 'stylo', 'contact' ou 'centre'")

    # roue dedans => centre de roue du côté opposé à la normale de la piste
    sign_side = -1.0 if relation == "dedans" else 1.0

    for i in range(steps):
        s = s_max * i / (steps - 1)
        C, theta, N_vec = _interpolate_on_segments(s % L, segments)
        x_track, y_track = C
        nx, ny = N_vec

        # centre de la roue
        cx = x_track + sign_side * nx * r_wheel
        cy = y_track + sign_side * ny * r_wheel

        if mode == "centre":
            base_points.append((cx, cy))
            continue

        # approximation : la longueur parcourue sur la piste en "dents" vaut
        # s / pitch_mm_per_tooth, avec un décalage initial optionnel.
        teeth_rolled = (s / pitch_mm_per_tooth) + float(wheel_phase_teeth)

        # Phase de la roue (sens horaire Spirograph)
        phi = -2.0 * math.pi * (teeth_rolled / float(N_w))

        if mode == "contact":
            base_points.append((x_track, y_track))
            continue

        px = cx + d * math.cos(phi)
        py = cy + d * math.sin(phi)

        base_points.append((px, py))

    return base_points
