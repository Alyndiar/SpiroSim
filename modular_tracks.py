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
from dataclasses import dataclass, field
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
    segments: List["TrackSegment"]  # segments analytiques pour retrouver la courbure locale
    pieces: List["PieceUsage"] = field(default_factory=list)


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


@dataclass
class PieceUsage:
    """Instance construite d'une pièce modulaire avec ses métriques utiles."""

    name: str
    sign: str
    kind: str  # "arc" ou "line"
    arc_degrees: Optional[float]
    length_mm: float
    teeth_equiv: float
    s_start: float
    s_end: float
    r_pitch: Optional[float] = None
    r_center: Optional[float] = None

# Segments analytiques de la piste (centre)
@dataclass
class TrackSegment:
    kind: str              # "arc" ou "line"
    s_start: float
    s_end: float
    side_sign: float = 0.0  # +1 = arc concave, -1 = arc convexe, 0 = segment droit
    piece_name: Optional[str] = None

    # rayon de pas utilisé pour l'engrènement (inner/outer selon le signe)
    r_pitch: float = 0.0

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
) -> Tuple[List[TrackSegment], List[PieceUsage], float, float]:
    """
    Construit des segments analytiques (arcs / lignes) pour la piste.

    Règles :
      - Le point de départ de la piste (début de la 1ʳᵉ pièce) est au (0, 0).
      - 1ʳᵉ pièce courbe :
            * convexe  (signe '-') -> centre de l’arc (0, R)
            * concave  (signe '+') -> centre de l’arc (0, -r)
        où R/r = rayon de la piste centrale (r_pitch_base).
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
    pieces: List[PieceUsage] = []
    s_cur = 0.0
    total_teeth_equiv = 0.0

    # Rayons inner / outer, puis rayon de pas (commun à toutes les pièces)
    if inner_teeth > 0:
        r_inner = inner_teeth * pitch_mm_per_tooth / (2.0 * math.pi)
    else:
        r_inner = 0.0
    if outer_teeth > 0:
        r_outer = outer_teeth * pitch_mm_per_tooth / (2.0 * math.pi)
    else:
        r_outer = r_inner

    if r_inner > 0 or r_outer > 0:
        r_pitch_base = 0.5 * (r_inner + r_outer)
    else:
        # valeur arbitraire non nulle, au cas où
        r_pitch_base = 10.0

    # point et tangente courants (pour les pièces > 1)
    # Orientation canonique :
    #   - on part du point (0, 0) à l'angle 0 (vers +X)
    #   - une rotation globale de +90° sera appliquée plus tard pour placer
    #     le départ en haut (π/2) pour l'affichage/usages externes.
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
        sign_char = elem.sign or "+"

        # -------------------------
        # PIÈCES DROITES (E, F, Z…)
        # -------------------------
        if pdef.arc_degrees is None:
            teeth_here = pdef.straight_teeth
            if teeth_here <= 0:
                continue
            L = teeth_here * pitch_mm_per_tooth

            if not first_piece_done:
                # 1ʳᵉ pièce droite : centrée sur (0, 0) et VERTICALE
                # (cohérent avec l'orientation à angle 0 avant rotation globale)
                x_center = 0.0
                y_center = 0.0

                # offset en X (translation) si demandé
                x_center += remaining_offset_mm
                remaining_offset_mm = 0.0

                x0_local = x_center
                x1_local = x_center
                y0_local = y_center - L / 2.0
                y1_local = y_center + L / 2.0

                seg = TrackSegment(
                    kind="line",
                    s_start=s_cur,
                    s_end=s_cur + L,
                    side_sign=0.0,
                    piece_name=pdef.name,
                    x0=x0_local,
                    y0=y0_local,
                    x1=x1_local,
                    y1=y1_local,
                    teeth_equiv=teeth_here,
                )
                segments.append(seg)

                pieces.append(
                    PieceUsage(
                        name=pdef.name,
                        sign=sign_char,
                        kind="line",
                        arc_degrees=None,
                        length_mm=L,
                        teeth_equiv=teeth_here,
                        s_start=s_cur,
                        s_end=s_cur + L,
                    )
                )

                s_cur += L
                total_teeth_equiv += teeth_here

                # pour la pièce suivante, on continue à partir de l’extrémité haute
                x0 = x1_local
                y0 = y1_local
                theta = math.pi / 2.0  # vers +Y pour rester cohérent avec angle 0

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
                    side_sign=0.0,
                    piece_name=pdef.name,
                    x0=x0,
                    y0=y0,
                    x1=x1,
                    y1=y1,
                    teeth_equiv=teeth_here,
                )
                segments.append(seg)

                pieces.append(
                    PieceUsage(
                        name=pdef.name,
                        sign=sign_char,
                        kind="line",
                        arc_degrees=None,
                        length_mm=L,
                        teeth_equiv=teeth_here,
                        s_start=s_cur,
                        s_end=s_cur + L,
                    )
                )

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

        # côté concave/convexe : signe +/-
        # + => concave (côté "intérieur")
        # - => convexe (côté "extérieur")
        side = 1.0
        if elem.sign == "-":
            side = -1.0

        # rayon de centre : dépend du côté actif (concave -> inner, convexe -> outer)
        if side > 0 and r_inner > 0.0:
            r_center = r_inner
        elif side < 0 and r_outer > 0.0:
            r_center = r_outer
        else:
            r_center = r_pitch_base

        # rayon de pas commun (affichage/roulage) : constant pour toute la piste
        r_pitch = r_pitch_base
        teeth_here = 0.0
        if r_pitch > 0.0:
            teeth_here = (abs(angle_deg) / 360.0) * (
                inner_teeth if side > 0 else outer_teeth
            )

        if teeth_here > 0.0:
            L_arc = abs(teeth_here * pitch_mm_per_tooth)
        else:
            L_arc = abs(r_center * angle)

        if not first_piece_done:
            # 1ʳᵉ pièce courbe : point de départ au (0, 0)
            # centre sur l'axe X pour se placer à angle 0 (droite)
            if elem.sign == "+":
                cx = -r_center
                alpha0 = 0.0  # vecteur centre->point = (r, 0)
            else:
                cx = r_center
                alpha0 = math.pi  # vecteur centre->point = (-r, 0)
            cy = 0.0

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
            side_sign=side,
            piece_name=pdef.name,
            cx=cx,
            cy=cy,
            r_center=r_center,
            r_pitch=r_pitch if r_pitch > 0.0 else r_center,
            angle_start=alpha0,
            angle_end=alpha1,
            teeth_equiv=teeth_here,
        )
        segments.append(seg)

        pieces.append(
            PieceUsage(
                name=pdef.name,
                sign=sign_char,
                kind="arc",
                arc_degrees=angle_deg,
                length_mm=L_arc,
                teeth_equiv=teeth_here,
                s_start=s_cur,
                s_end=s_cur + L_arc,
                r_pitch=r_pitch if r_pitch > 0.0 else None,
                r_center=r_center,
            )
        )

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
    return segments, pieces, total_length, total_teeth_equiv

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
    # Pièce Y : jonction triple, arcs concaves de 120° (orientation gérée ailleurs)
    "Y": PieceDef("Y", arc_degrees=120.0, straight_teeth=0.0),
    "Z": PieceDef("Z", arc_degrees=None, straight_teeth=14.0),  # end piece
}


def describe_track_pieces(track: TrackBuildResult) -> List[str]:
    """Renvoie une description textuelle des pièces utilisées pour une piste."""

    descriptions: List[str] = []
    for idx, p in enumerate(track.pieces, start=1):
        face = "concave/intérieur" if p.sign == "+" else "convexe/extérieur"
        teeth_txt = f"{p.teeth_equiv:.3f} dents équiv."
        length_txt = f"{p.length_mm:.2f} mm"

        if p.kind == "arc":
            arc_txt = f"{p.arc_degrees:g}°" if p.arc_degrees is not None else "?°"
            pitch_txt = f", r_pitch={p.r_pitch:.3f} mm" if p.r_pitch else ""
            center_txt = f", r_centre={p.r_center:.3f} mm" if p.r_center else ""
            desc = (
                f"{idx:02d}. {p.sign}{p.name} (arc {arc_txt}, {face}): "
                f"{teeth_txt}, {length_txt}{pitch_txt}{center_txt}"
            )
        else:
            desc = f"{idx:02d}. {p.sign}{p.name} (barre): {teeth_txt}, {length_txt}"

        descriptions.append(desc)

    return descriptions


def describe_notation(
    notation: str,
    inner_teeth: int = 96,
    outer_teeth: int = 144,
    pitch_mm_per_tooth: float = 0.65,
) -> List[str]:
    """
    Construit la piste puis retourne la liste des pièces décrites texte.

    Utile pour déboguer un tracé sans passer par l'interface graphique.
    """

    track = build_track_from_notation(
        notation,
        inner_teeth=inner_teeth,
        outer_teeth=outer_teeth,
        pitch_mm_per_tooth=pitch_mm_per_tooth,
        steps_per_tooth=1,
    )
    return describe_track_pieces(track)


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
        * recentrer sur le barycentre puis appliquer une rotation
          globale de +90° pour positionner le départ à π/2.
    """

    (
        segments,
        pieces,
        total_length,
        total_teeth_equiv,
    ) = build_segments_for_parsed_track(
        parsed,
        inner_teeth=inner_teeth,
        outer_teeth=outer_teeth,
        pitch_mm_per_tooth=pitch_mm_per_tooth,
        offset_teeth=parsed.offset_teeth,   # offset utilisé dans les segments
    )

    points = segments_to_polyline(segments, steps_per_tooth)

    if not points:
        points = [(0.0, 0.0)]

    # --- Recentrer sur le barycentre ---
    cx = sum(p[0] for p in points) / len(points)
    cy = sum(p[1] for p in points) / len(points)
    points = [(x - cx, y - cy) for (x, y) in points]

    # --- Rotation globale de +90° pour aligner le départ à π/2 ---
    rot = math.pi / 2.0
    cos_r = math.cos(rot)
    sin_r = math.sin(rot)
    points = [(x * cos_r - y * sin_r, x * sin_r + y * cos_r) for (x, y) in points]

    def _transform_point(x: float, y: float) -> Tuple[float, float]:
        """Applique la translation barycentrique puis la rotation globale."""

        x -= cx
        y -= cy
        xr = x * cos_r - y * sin_r
        yr = x * sin_r + y * cos_r
        return xr, yr

    def _rotate_angle(a: float) -> float:
        return a + rot

    # Appliquer la même transform aux segments pour garder cohérence
    for seg in segments:
        if seg.kind == "line":
            seg.x0, seg.y0 = _transform_point(seg.x0, seg.y0)
            seg.x1, seg.y1 = _transform_point(seg.x1, seg.y1)
        else:
            seg.cx, seg.cy = _transform_point(seg.cx, seg.cy)
            seg.angle_start = _rotate_angle(seg.angle_start)
            seg.angle_end = _rotate_angle(seg.angle_end)

    return TrackBuildResult(
        points=points,
        total_length=total_length,
        total_teeth=total_teeth_equiv,
        offset_teeth=parsed.offset_teeth,
        segments=segments,
        pieces=pieces,
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
        return TrackBuildResult(
            points=[(0.0, 0.0)],
            total_length=0.0,
            total_teeth=0.0,
            offset_teeth=parsed.offset_teeth,
            segments=[],
            pieces=[],
        )
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


def _interpolate_on_segments(
    s: float, track: TrackBuildResult
) -> Tuple[float, float, float, Optional[TrackSegment], float]:
    """
    Interpolation analytique basée sur les segments, avec récupération du segment local
    et de la courbure (0 pour une barre, ±1/r pour un arc).
    """
    if not track.segments:
        cum, tangents = _precompute_length_and_tangent(track.points)
        x, y, theta = _interpolate_on_track(s, track.points, cum, tangents)
        return x, y, theta, None, 0.0

    L = track.total_length
    if L <= 0:
        seg = track.segments[0]
        if seg.kind == "line":
            theta = math.atan2(seg.y1 - seg.y0, seg.x1 - seg.x0)
            return seg.x0, seg.y0, theta, seg, 0.0
        angle = seg.angle_start
        x = seg.cx + seg.r_center * math.cos(angle)
        y = seg.cy + seg.r_center * math.sin(angle)
        theta = angle + seg.side_sign * (math.pi / 2.0)
        curvature = seg.side_sign / seg.r_center if seg.r_center else 0.0
        return x, y, theta, seg, curvature

    s_mod = s % L
    seg = track.segments[-1]
    for candidate in track.segments:
        if s_mod <= candidate.s_end:
            seg = candidate
            break

    if seg.s_end <= seg.s_start:
        if seg.kind == "line":
            theta = math.atan2(seg.y1 - seg.y0, seg.x1 - seg.x0)
            return seg.x0, seg.y0, theta, seg, 0.0
        angle = seg.angle_start
        x = seg.cx + seg.r_center * math.cos(angle)
        y = seg.cy + seg.r_center * math.sin(angle)
        theta = angle + seg.side_sign * (math.pi / 2.0)
        curvature = seg.side_sign / seg.r_center if seg.r_center else 0.0
        return x, y, theta, seg, curvature

    t = (s_mod - seg.s_start) / (seg.s_end - seg.s_start)
    if seg.kind == "line":
        x = seg.x0 + (seg.x1 - seg.x0) * t
        y = seg.y0 + (seg.y1 - seg.y0) * t
        theta = math.atan2(seg.y1 - seg.y0, seg.x1 - seg.x0)
        return x, y, theta, seg, 0.0

    angle = seg.angle_start + (seg.angle_end - seg.angle_start) * t
    x = seg.cx + seg.r_center * math.cos(angle)
    y = seg.cy + seg.r_center * math.sin(angle)
    theta = angle + seg.side_sign * (math.pi / 2.0)
    curvature = seg.side_sign / seg.r_center if seg.r_center else 0.0
    return x, y, theta, seg, curvature


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
    output_mode: str = "stylo",
    wheel_phase_teeth: float = 0.0,
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
    output_mode :
      - "stylo" (défaut) : renvoie le tracé du stylo.
      - "contact" : renvoie le point de contact (centre de la piste).
      - "centre" : renvoie la position du centre de la roue.
    wheel_phase_teeth :
      - décalage initial (en dents) de la roue par rapport au point 0 de la piste.
        (positif = roue avancée vers la droite du point 0, négatif = vers la gauche)
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

    cum: List[float] = []
    tangents: List[float] = []
    if not track.segments:
        cum, tangents = _precompute_length_and_tangent(track.points)

    L = track.total_length if track.total_length > 0 else (cum[-1] if cum else 0.0)

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

    mode = output_mode.lower()
    if mode not in {"stylo", "contact", "centre"}:
        raise ValueError("output_mode doit être 'stylo', 'contact' ou 'centre'")

    def normal_multiplier(seg: Optional[TrackSegment]) -> float:
        """Choisit le côté de la normale selon la relation et la concavité locale."""
        if seg is None or seg.kind != "arc" or seg.side_sign == 0:
            return -1.0 if relation == "dedans" else 1.0
        # concave (+) : normale gauche pointe vers le centre ; convexe (-) : gauche pointe vers l'extérieur
        return seg.side_sign if relation == "dedans" else -seg.side_sign

    def rolling_mode(seg: Optional[TrackSegment]) -> str:
        if seg is None or seg.kind == "line":
            return "cycloide"
        if seg.side_sign >= 0:
            return "hypotrochoide" if relation == "dedans" else "epitrochoide"
        return "epitrochoide" if relation == "dedans" else "hypotrochoide"

    def rolling_factor(seg: Optional[TrackSegment]) -> float:
        """Retourne dphi/ds pour le segment courant."""
        if seg is None:
            return -1.0 / r_wheel
        if seg.kind == "line":
            return -1.0 / r_wheel
        R = max(seg.r_pitch if seg.r_pitch > 0.0 else seg.r_center, 1e-9)
        mode = rolling_mode(seg)
        if mode == "hypotrochoide":
            return -(R - r_wheel) / (r_wheel * R)
        return -(R + r_wheel) / (r_wheel * R)

    # Pré-calcul de la contribution d'une boucle complète pour rendre phi continu
    phi_per_lap = 0.0
    for seg in track.segments:
        dphi_ds = rolling_factor(seg)
        phi_per_lap += dphi_ds * max(0.0, seg.s_end - seg.s_start)

    phase_rad = -2.0 * math.pi * (float(wheel_phase_teeth) / float(wt))

    for i in range(steps):
        s = s_max * i / (steps - 1)
        if not track.segments:
            x_track, y_track, theta = _interpolate_on_track(s, track.points, cum, tangents)
            teeth_rolled = (s / pitch_mm_per_tooth) + float(wheel_phase_teeth)
            phi = -2.0 * math.pi * (teeth_rolled / float(wt))

            # normale selon la relation et la concavité (fallback gauche/droite par défaut)
            n_mult = normal_multiplier(None)
            nx = n_mult * -math.sin(theta)
            ny = n_mult * math.cos(theta)

            cx = x_track + nx * r_wheel
            cy = y_track + ny * r_wheel

            if mode == "centre":
                base_points.append((cx, cy))
                continue

            if mode == "contact":
                base_points.append((x_track, y_track))
                continue

            px = cx + d * math.cos(phi)
            py = cy + d * math.sin(phi)
            base_points.append((px, py))
            continue
        laps = 0
        if L > 0:
            laps = int(s // L)
        s_local = s - laps * L if L > 0 else 0.0

        x_track, y_track, theta, seg, _ = _interpolate_on_segments(s_local, track)

        # Calcul de phi en intégrant dphi/ds sur les segments
        phi = phase_rad + phi_per_lap * laps
        remaining = s_local
        for seg_piece in track.segments:
            seg_len = max(0.0, seg_piece.s_end - seg_piece.s_start)
            if remaining > seg_len:
                phi += rolling_factor(seg_piece) * seg_len
                remaining -= seg_len
            else:
                phi += rolling_factor(seg_piece) * remaining
                break

        # normale selon la relation et la concavité locale
        n_mult = normal_multiplier(seg)
        nx = n_mult * -math.sin(theta)
        ny = n_mult * math.cos(theta)

        # centre de la roue : dedans/dehors selon sign_side
        cx = x_track + nx * r_wheel
        cy = y_track + ny * r_wheel

        if mode == "centre":
            base_points.append((cx, cy))
            continue

        if mode == "contact":
            base_points.append((x_track, y_track))
            continue

        px = cx + d * math.cos(phi)
        py = cy + d * math.sin(phi)

        base_points.append((px, py))

    return base_points
