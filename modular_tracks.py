"""
modular_tracks.py

Génération de pistes modulaires de type SuperSpirograph + courbes associées.

Ce module est indépendant de Qt. Il fournit :
  - un parseur pour la notation de piste (offset +A -B * etc.)
  - la construction d'une polyline centrale représentant la piste
  - la génération d'un tracé de stylo pour une roue donnée qui roule
    le long de cette piste, côté "dedans" ou "dehors".

Hypothèses / simplifications :
  - Toutes les pièces courbes (A, B, C, D, Y, ...) sont définies par
    un angle en degrés (45, 60, 90, 120, ...).
  - On suppose un pas constant par dent (pitch_mm_per_tooth).
  - Pour un anneau avec inner_teeth / outer_teeth, le nombre de dents
    "théorique" utilisé par un arc est :
        T = inner_teeth * angle / 360  pour le côté concave (+)
        T = outer_teeth * angle / 360  pour le côté convexe (-)
    Ce T peut être non entier pour certains anneaux (ex : 105 dents),
    ce qui est acceptable pour le dessin numérique.
  - La pièce Y (jonction triple) est reconnue dans la notation mais
    pas encore géométriquement implémentée (NotImplementedError).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple, Optional

Point = Tuple[float, float]


# ---------------------------------------------------------------------------
# 1) Structures de données
# ---------------------------------------------------------------------------

@dataclass
class TrackBuildResult:
    """Résultat de la construction d'une piste modulaire."""

    points: List[Point]      # polyline centrale de la piste (mm)
    total_length: float      # longueur totale (mm)
    total_teeth: float       # nombre "équivalent" de dents total (somme des pièces)
    offset_teeth: int        # décalage initial (en dents)


@dataclass
class PieceDef:
    """
    Définition d'un type de pièce modulaire.

    arc_degrees :
      - angle de l'arc en degrés pour les pièces courbes (A, B, C, D, Y, ...)
      - None pour les pièces droites (E, F, Z)
    straight_teeth :
      - nombre de dents de rack pour les pièces droites (barres)
      - 0 ou None pour les pièces courbes
    """

    name: str
    arc_degrees: Optional[float] = None
    straight_teeth: float = 0.0

# Segments analytiques de la piste (centre)
@dataclass
class TrackSegment:
    kind: str              # "arc" ou "line"
    s_start: float
    s_end: float

    # arcs
    cx: float = 0.0
    cy: float = 0.0
    r_center: float = 0.0
    angle_start: float = 0.0
    angle_end: float = 0.0

    # segments droits
    x0: float = 0.0
    y0: float = 0.0
    x1: float = 0.0
    y1: float = 0.0

    teeth_equiv: float = 0.0


def build_segments_for_parsed_track(
    parsed: ParsedTrack,
    inner_teeth: float,
    outer_teeth: float,
    pitch_mm_per_tooth: float,
    offset_teeth: float = 0.0,
) -> Tuple[List[TrackSegment], float, float]:
    """
    Construit des segments analytiques (arcs / lignes) pour la piste.

    Règles :
      - Le point de départ de la piste (début de la 1ʳᵉ pièce) est au (0, 0).
      - 1ʳᵉ pièce courbe :
            * convexe  (signe '-') -> centre de l’arc (0, R)
            * concave  (signe '+') -> centre de l’arc (0, -r)
        où R/r = rayon de la piste centrale (r_center_base).
      - 1ʳᵉ pièce droite (barre E/F/Z) :
            * centrée sur (0, 0) : de -L/2 à +L/2 sur l’axe X.

      - L’offset (en dents) est appliqué UNIQUEMENT sur cette 1ʳᵉ pièce :
            * barre : translation en X de offset_mm
            * arc   : rotation le long de l’arc, autour de son centre
                      (offset > 0 :
                         concave  => sens antihoraire (CCW)
                         convexe  => sens horaire (CW) )
    """

    segments: List[TrackSegment] = []
    s_cur = 0.0
    total_teeth_equiv = 0.0

    # Rayons inner / outer, puis rayon de la piste centrale
    if inner_teeth > 0:
        r_inner = inner_teeth * pitch_mm_per_tooth / (2.0 * math.pi)
    else:
        r_inner = 0.0
    if outer_teeth > 0:
        r_outer = outer_teeth * pitch_mm_per_tooth / (2.0 * math.pi)
    else:
        r_outer = r_inner

    if r_inner > 0 or r_outer > 0:
        r_center_base = 0.5 * (r_inner + r_outer)
    else:
        # valeur arbitraire non nulle, au cas où
        r_center_base = 10.0

    # point et tangente courants (pour les pièces > 1)
    x0 = 0.0
    y0 = 0.0
    theta = 0.0  # direction initiale (vers +X)

    # offset global, en mm, à consommer sur la première pièce
    remaining_offset_mm = float(offset_teeth) * pitch_mm_per_tooth
    first_piece_done = False

    for elem in parsed.elements:
        if elem.kind != "piece":
            continue

        pdef = PIECES[elem.piece_name]

        # -------------------------
        # PIÈCES DROITES (E, F, Z…)
        # -------------------------
        if pdef.arc_degrees is None:
            teeth_here = pdef.straight_teeth
            if teeth_here <= 0:
                continue
            L = teeth_here * pitch_mm_per_tooth

            if not first_piece_done:
                # 1ʳᵉ pièce droite : centrée sur (0, 0)
                x_center = 0.0
                y_center = 0.0

                # offset en X (translation) si demandé
                x_center += remaining_offset_mm
                remaining_offset_mm = 0.0

                x0_local = x_center - L / 2.0
                x1_local = x_center + L / 2.0
                y0_local = y_center
                y1_local = y_center

                seg = TrackSegment(
                    kind="line",
                    s_start=s_cur,
                    s_end=s_cur + L,
                    x0=x0_local,
                    y0=y0_local,
                    x1=x1_local,
                    y1=y1_local,
                    teeth_equiv=teeth_here,
                )
                segments.append(seg)

                s_cur += L
                total_teeth_equiv += teeth_here

                # pour la pièce suivante, on continue à partir de l’extrémité droite
                x0 = x1_local
                y0 = y1_local
                theta = 0.0  # toujours vers +X ici

                first_piece_done = True
            else:
                # pièces droites suivantes : continuation à partir de (x0, y0, theta)
                vx = math.cos(theta)
                vy = math.sin(theta)
                x1 = x0 + vx * L
                y1 = y0 + vy * L

                seg = TrackSegment(
                    kind="line",
                    s_start=s_cur,
                    s_end=s_cur + L,
                    x0=x0,
                    y0=y0,
                    x1=x1,
                    y1=y1,
                    teeth_equiv=teeth_here,
                )
                segments.append(seg)

                s_cur += L
                total_teeth_equiv += teeth_here
                x0, y0 = x1, y1
                # theta inchangé pour une barre

            continue  # fin du traitement des pièces droites

        # -------------------------
        # PIÈCES COURBES (A, B, C, D, Y…)
        # -------------------------
        angle_deg = pdef.arc_degrees
        angle = math.radians(angle_deg)
        if angle == 0.0:
            continue

        r_center = r_center_base
        L_arc = abs(r_center * angle)
        teeth_here = L_arc / pitch_mm_per_tooth

        # côté concave/convexe : signe +/-
        # + => concave (côté "intérieur")
        # - => convexe (côté "extérieur")
        side = 1.0
        if elem.sign == "-":
            side = -1.0

        if not first_piece_done:
            # 1ʳᵉ pièce courbe : point de départ au (0, 0)
            # et centre à (0, R) pour convexe, (0, -r) pour concave.
            cx = 0.0
            cy = r_center if elem.sign == "-" else -r_center

            # angle de départ pour que le point (0,0) soit sur l’arc
            # concave (centre en (0, -r))  -> angle +π/2 donne (0,0)
            # convexe (centre en (0,  r))  -> angle -π/2 donne (0,0)
            alpha0 = math.pi / 2.0 if elem.sign == "+" else -math.pi / 2.0

            # --- appliquer l’offset sur cette 1ʳᵉ pièce courbe ---
            if remaining_offset_mm != 0.0:
                # on limite à un tour d’arc pour éviter les grands tours
                s_off = math.fmod(remaining_offset_mm, L_arc)
                if s_off == 0.0 and remaining_offset_mm != 0.0:
                    s_off = remaining_offset_mm

                # conversion longueur -> angle le long de l’arc
                # règle :
                #   - concave  (+) : offset>0 => antihoraire (CCW)
                #   - convexe  (-) : offset>0 => horaire (CW)
                # on obtient cela avec : delta_alpha = side * (s_off / r_center)
                delta_alpha = side * (s_off / r_center)
                alpha0 = alpha0 + delta_alpha

                remaining_offset_mm = 0.0
            # ------------------------------------------------------

            # point de départ effectif sur l’arc
            x0 = cx + r_center * math.cos(alpha0)
            y0 = cy + r_center * math.sin(alpha0)

        else:
            # pièces courbes suivantes : centre en fonction de (x0, y0, theta)
            nx = -math.sin(theta)
            ny = math.cos(theta)
            cx = x0 + side * nx * r_center
            cy = y0 + side * ny * r_center

            rx0 = x0 - cx
            ry0 = y0 - cy
            alpha0 = math.atan2(ry0, rx0)

        # paramètres de l’arc
        alpha1 = alpha0 + side * angle

        seg = TrackSegment(
            kind="arc",
            s_start=s_cur,
            s_end=s_cur + L_arc,
            cx=cx,
            cy=cy,
            r_center=r_center,
            angle_start=alpha0,
            angle_end=alpha1,
            teeth_equiv=teeth_here,
        )
        segments.append(seg)

        s_cur += L_arc
        total_teeth_equiv += teeth_here

        # fin de l’arc : nouveau point
        x1 = cx + r_center * math.cos(alpha1)
        y1 = cy + r_center * math.sin(alpha1)
        x0, y0 = x1, y1

        # nouvelle tangente à la fin de l’arc
        tx = -side * math.sin(alpha1)
        ty =  side * math.cos(alpha1)
        theta = math.atan2(ty, tx)

        first_piece_done = True

    total_length = s_cur
    return segments, total_length, total_teeth_equiv

def segments_to_polyline(segments: List[TrackSegment], steps_per_tooth: int) -> List[Point]:
    """
    Échantillonne une liste de segments analytiques en polyline centrale.
    """
    points: List[Point] = []

    for seg in segments:
        if seg.s_end <= seg.s_start:
            continue

        if seg.kind == "line":
            # nombre de points basé sur les dents équivalentes
            n_steps = max(2, int(max(1.0, seg.teeth_equiv) * steps_per_tooth))
            for i in range(n_steps):
                t = i / float(n_steps - 1)
                x = seg.x0 + (seg.x1 - seg.x0) * t
                y = seg.y0 + (seg.y1 - seg.y0) * t
                points.append((x, y))

        elif seg.kind == "arc":
            if seg.r_center == 0.0:
                continue
            n_steps = max(4, int(max(1.0, seg.teeth_equiv) * steps_per_tooth))
            d_angle = seg.angle_end - seg.angle_start
            for i in range(n_steps):
                t = i / float(n_steps - 1)
                a = seg.angle_start + d_angle * t
                x = seg.cx + seg.r_center * math.cos(a)
                y = seg.cy + seg.r_center * math.sin(a)
                points.append((x, y))

    if not points:
        points = [(0.0, 0.0)]
    return points

# Angles approximatifs pour les pièces courbes
# E/F/Z : barres (ou fin de piste) définies par un nombre de dents "de rack"
PIECES = {
    "A": PieceDef("A", arc_degrees=45.0),
    "B": PieceDef("B", arc_degrees=60.0),
    "C": PieceDef("C", arc_degrees=90.0),
    "D": PieceDef("D", arc_degrees=120.0),
    "E": PieceDef("E", arc_degrees=None, straight_teeth=20.0),
    "F": PieceDef("F", arc_degrees=None, straight_teeth=56.0),
    "Y": PieceDef("Y", arc_degrees=60.0, straight_teeth=0.0),  # jonction triple
    "Z": PieceDef("Z", arc_degrees=None, straight_teeth=14.0),  # end piece
}


@dataclass
class ParsedElement:
    kind: str              # "piece" ou "branch"
    sign: Optional[str] = None   # "+" ou "-"
    piece_name: Optional[str] = None


@dataclass
class ParsedTrack:
    offset_teeth: int
    elements: List[ParsedElement]


# ---------------------------------------------------------------------------
# 2) Parsing de la notation
# ---------------------------------------------------------------------------

def parse_track_notation(text: str) -> ParsedTrack:
    """
    Parse une notation du type : -18-C+D+B-C+D+...

    offset_teeth :
      - entier signé (peut être 0 ou absent)
    éléments :
      - "+X" ou "-X" pour les pièces (X = A, B, C, D, E, F, Y, Z)
      - "*" pour sauter de branche sur les pièces spéciales (pour l'instant
        simplement enregistré comme "branch" et ignoré côté géométrie).
    """
    s = text.strip()
    if not s:
        return ParsedTrack(0, [])

    idx = 0
    n = len(s)

    # 1) décalage initial (entier signé, optionnel)
    # On ne le prend en compte que si l'on trouve réellement des chiffres.
    offset_teeth = 0
    orig_idx = idx
    sign = 1

    if idx < n and s[idx] in "+-":
        # On regarde si ce signe est suivi de chiffres : sinon, ce n’est pas un offset.
        sign_char = s[idx]
        idx += 1
        if idx < n and s[idx].isdigit():
            sign = -1 if sign_char == "-" else 1
            start_idx = idx
            while idx < n and s[idx].isdigit():
                idx += 1
            offset_teeth = sign * int(s[start_idx:idx])
        else:
            # Pas de chiffres après le signe : ce n'était pas un offset,
            # on revient au début pour laisser le signe au premier élément.
            idx = orig_idx
    elif idx < n and s[idx].isdigit():
        # Offset sans signe explicite, ex: "18-C+D+..."
        start_idx = idx
        while idx < n and s[idx].isdigit():
            idx += 1
        offset_teeth = int(s[start_idx:idx])

    elements: List[ParsedElement] = []

    # 2) suite d’opérateurs (+A, -C, *, ...)
    while idx < n:
        ch = s[idx]
        if ch.isspace():
            idx += 1
            continue
        if ch == "*":
            elements.append(ParsedElement(kind="branch"))
            idx += 1
            continue
        if ch in "+-":
            sig = ch
            idx += 1
            if idx >= n:
                break
            piece_name = s[idx].upper()
            idx += 1
            if piece_name not in PIECES:
                raise ValueError(f"Pièce inconnue dans la notation : {piece_name!r}")
            elements.append(ParsedElement(kind="piece", sign=sig, piece_name=piece_name))
            continue
        raise ValueError(f"Caractère inattendu dans la notation : {ch!r} à la position {idx}")

    return ParsedTrack(offset_teeth=offset_teeth, elements=elements)


# ---------------------------------------------------------------------------
# 3) Construction de la polyline de piste
# ---------------------------------------------------------------------------

def _build_polyline_for_parsed_track(
    parsed: ParsedTrack,
    inner_teeth: int = 96,
    outer_teeth: int = 144,
    pitch_mm_per_tooth: float = 0.65,
    steps_per_tooth: int = 3,
) -> TrackBuildResult:
    """
    Construit la polyline centrale pour une piste modulaire, à partir
    des segments analytiques.

    - L'offset (en dents) est appliqué LOCALLEMENT dans
      build_segments_for_parsed_track sur la première pièce courbe.
    - Ici, on se contente de :
        * générer les segments,
        * les échantillonner en polyligne,
        * recentrer sur le barycentre.

    AUCUNE rotation globale, AUCUNE orientation canonique.
    """

    segments, total_length, total_teeth_equiv = build_segments_for_parsed_track(
        parsed,
        inner_teeth=inner_teeth,
        outer_teeth=outer_teeth,
        pitch_mm_per_tooth=pitch_mm_per_tooth,
        offset_teeth=parsed.offset_teeth,   # offset utilisé dans les segments
    )

    points = segments_to_polyline(segments, steps_per_tooth)

    if not points:
        points = [(0.0, 0.0)]

    # --- Recentrer sur le barycentre (UNIQUEMENT ce recentrage) ---
    cx = sum(p[0] for p in points) / len(points)
    cy = sum(p[1] for p in points) / len(points)
    points = [(x - cx, y - cy) for (x, y) in points]

    return TrackBuildResult(
        points=points,
        total_length=total_length,
        total_teeth=total_teeth_equiv,
        offset_teeth=parsed.offset_teeth,
    )

def build_track_from_notation(
    notation: str,
    inner_teeth: int = 96,
    outer_teeth: int = 144,
    pitch_mm_per_tooth: float = 0.65,
    steps_per_tooth: int = 3,
) -> TrackBuildResult:
    """Helper direct : parse puis construit la polyline pour une piste."""
    parsed = parse_track_notation(notation)
    if not parsed.elements:
        return TrackBuildResult(points=[(0.0, 0.0)], total_length=0.0, total_teeth=0.0, offset_teeth=parsed.offset_teeth)
    return _build_polyline_for_parsed_track(
        parsed,
        inner_teeth=inner_teeth,
        outer_teeth=outer_teeth,
        pitch_mm_per_tooth=pitch_mm_per_tooth,
        steps_per_tooth=steps_per_tooth,
    )


# ---------------------------------------------------------------------------
# 4) Outils d'interpolation le long de la piste
# ---------------------------------------------------------------------------

def _precompute_length_and_tangent(points: List[Point]):
    """Prépare longueurs cumulées + angles de tangente pour une polyline."""
    n = len(points)
    if n < 2:
        return [0.0], [0.0]

    cum = [0.0]
    tangents = [0.0]
    total = 0.0
    for i in range(1, n):
        x0, y0 = points[i - 1]
        x1, y1 = points[i]
        dx = x1 - x0
        dy = y1 - y0
        seg_len = math.hypot(dx, dy)
        if seg_len <= 0:
            tangents.append(tangents[-1])
            cum.append(total)
            continue
        total += seg_len
        cum.append(total)
        tangents.append(math.atan2(dy, dx))
    return cum, tangents


def _interpolate_on_track(
    s: float,
    points: List[Point],
    cum: List[float],
    tangents: List[float],
) -> Tuple[float, float, float]:
    """Renvoie (x, y, theta) pour une abscisse curviligne s (modulo la longueur)."""
    if not points:
        return 0.0, 0.0, 0.0

    L = cum[-1]
    if L <= 0:
        x, y = points[0]
        return x, y, 0.0

    # on boucle
    s = s % L

    # recherche linéaire (suffisant pour ~quelques milliers de points)
    i = 1
    n = len(cum)
    while i < n and cum[i] < s:
        i += 1
    if i >= n:
        i = n - 1

    s0 = cum[i - 1]
    s1 = cum[i]
    if s1 <= s0:
        x, y = points[i]
        return x, y, tangents[i]

    t = (s - s0) / (s1 - s0)
    x0, y0 = points[i - 1]
    x1, y1 = points[i]
    x = x0 + (x1 - x0) * t
    y = y0 + (y1 - y0) * t
    theta = tangents[i]
    return x, y, theta


# ---------------------------------------------------------------------------
# 5) Génération de la courbe trochoïdale le long d'une piste
# ---------------------------------------------------------------------------

def generate_track_base_points(
    notation: str,
    wheel_teeth: int,
    hole_index: float,
    hole_spacing_mm: float,
    steps: int,
    relation: str = "dedans",
    inner_teeth: int = 96,
    outer_teeth: int = 144,
    pitch_mm_per_tooth: float = 0.65,
) -> List[Point]:
    """
    Génère une courbe trochoïdale le long d'une piste modulaire.

    notation :
      - chaîne décrivant la piste (ex: "-18-C+D+B-C+D+...").
    wheel_teeth :
      - nombre de dents de la roue mobile.
    hole_index :
      - index du trou (comme dans Gears.py).
    relation :
      - "dedans" : la roue roule côté intérieur de la piste.
      - "dehors" : la roue roule côté extérieur.
    inner_teeth / outer_teeth :
      - paramètres d'anneau sur lequel sont basées les pièces courbes.
    pitch_mm_per_tooth :
      - longueur d'arc correspondant à une dent.
    """
    if steps < 2:
        steps = 2

    track = build_track_from_notation(
        notation,
        inner_teeth=inner_teeth,
        outer_teeth=outer_teeth,
        pitch_mm_per_tooth=pitch_mm_per_tooth,
        steps_per_tooth=3,
    )
    if len(track.points) < 2 or track.total_length <= 0.0:
        return []

    # Rayon du cercle de pas de la roue mobile
    wt = max(1, int(wheel_teeth))
    r_wheel = (pitch_mm_per_tooth * float(wt)) / (2.0 * math.pi)

    # Distance stylo->centre de la roue
    R_tip = r_wheel
    d = R_tip - hole_index * hole_spacing_mm
    if d < 0:
        d = 0.0

    cum, tangents = _precompute_length_and_tangent(track.points)
    L = cum[-1]

    # Nombre "équivalent" de dents pour la piste
    if track.total_teeth > 0:
        N_track = max(1, int(round(track.total_teeth)))
    else:
        N_track = wt

    N_w = wt
    g = math.gcd(N_track, N_w)
    if g <= 0:
        g = 1
    nb_laps = N_w // g if N_w >= g else 1
    if nb_laps < 1:
        nb_laps = 1

    s_max = L * float(nb_laps)

    base_points: List[Point] = []

    sign_side = -1.0 if relation == "dedans" else 1.0

    for i in range(steps):
        s = s_max * i / (steps - 1)
        x_track, y_track, theta = _interpolate_on_track(s, track.points, cum, tangents)

        # normale vers la gauche
        nx = -math.sin(theta)
        ny = math.cos(theta)

        # centre de la roue : dedans/dehors selon sign_side
        cx = x_track + sign_side * nx * r_wheel
        cy = y_track + sign_side * ny * r_wheel

        # approximation : on considère que la longueur parcourue sur la piste
        # en dents est s / pitch_mm_per_tooth (s étant la distance curviligne).
        teeth_rolled = s / pitch_mm_per_tooth
        phi = -2.0 * math.pi * (teeth_rolled / float(wt))

        px = cx + d * math.cos(phi)
        py = cy + d * math.sin(phi)

        base_points.append((px, py))

    return base_points
