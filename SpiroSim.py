import importlib
import importlib.util
import sys
import math
import copy
import json
import re
import colorsys
import time
import os
import subprocess
from pathlib import Path
from html import escape  # <-- AJOUT ICI
from generated_colors import COLOR_NAME_TO_HEX
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import modular_tracks_2 as modular_tracks
import modular_tracks_2_demo as modular_track_demo
from shape_lab import ShapeDesignLabWindow
from shape_dsl import DslParseError, parse_analytic_expression, parse_modular_expression
from shape_geometry import (
    BaseCurve,
    ArcSegment,
    CircleCurve,
    EllipseCurve,
    LineSegment,
    ModularTrackCurve,
    build_circle,
    build_drop,
    build_oblong,
    build_rounded_polygon,
    pen_position,
)
import localisation
from localisation import gear_type_label, relation_label, tr

from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QDialog,
    QTreeWidget,
    QTreeWidgetItem,
    QFormLayout,
    QLineEdit,
    QDoubleSpinBox,
    QSpinBox,
    QCheckBox,
    QMessageBox,
    QLabel,
    QComboBox,
    QMenuBar,
    QMenu,
    QFileDialog,
    QSlider,          # <-- AJOUT
    QListWidget,      # <-- AJOUT
    QListWidgetItem,  # <-- AJOUT
    QStyle,
    QSizePolicy,
)
from PySide6.QtGui import (
    QAction,
    QPainter,
    QColor,
    QImage,
    QIcon,
    QFont,
    QPixmap,
    QDesktopServices,
    QPen,   # <-- AJOUT ICI
)
from PySide6.QtCore import (
    QByteArray, 
    Qt,
    Signal,
    QPoint,
    QPointF,
    QSize,
    QUrl,
    QTimer,
    QStandardPaths,
)
from PySide6.QtSvgWidgets import QSvgWidget
from PySide6.QtSvg import QSvgRenderer   # <-- AJOUTÉ

# ----- Constante : unités normalisées -----
# Les tailles et distances sont désormais exprimées en unités abstraites,
# sans conversion réelle.
UNIT_LENGTH = 1.0
def _load_app_version() -> str:
    spec = importlib.util.find_spec("spirosim._version")
    if spec is None:
        return "0.0.0-dev"
    version_module = importlib.import_module("spirosim._version")
    return getattr(version_module, "__version__", "0.0.0-dev")


APP_VERSION = _load_app_version()
GITHUB_REPO_URL = "https://github.com/alyndiar/SpiroSim"

def split_valid_modular_notation(text: str) -> Tuple[str, str, bool]:
    return modular_tracks.split_valid_modular_notation(text)

def wavelength_to_rgb(nm: float) -> Tuple[int, int, int]:
    """
    Convertit une longueur d'onde en nm (≈ 380–780) en RGB sRGB 0–255.
    Approximation standard (gamma 0.8).
    """
    w = float(nm)
    if w < 380 or w > 780:
        return (0, 0, 0)

    if 380 <= w < 440:
        r = -(w - 440.0) / (440.0 - 380.0)
        g = 0.0
        b = 1.0
    elif 440 <= w < 490:
        r = 0.0
        g = (w - 440.0) / (490.0 - 440.0)
        b = 1.0
    elif 490 <= w < 510:
        r = 0.0
        g = 1.0
        b = -(w - 510.0) / (510.0 - 490.0)
    elif 510 <= w < 580:
        r = (w - 510.0) / (580.0 - 510.0)
        g = 1.0
        b = 0.0
    elif 580 <= w < 645:
        r = 1.0
        g = -(w - 645.0) / (645.0 - 580.0)
        b = 0.0
    else:  # 645–780
        r = 1.0
        g = 0.0
        b = 0.0

    # facteur de sensibilité de l'œil
    if 380 <= w < 420:
        factor = 0.3 + 0.7 * (w - 380.0) / (420.0 - 380.0)
    elif 420 <= w < 700:
        factor = 1.0
    else:  # 700–780
        factor = 0.3 + 0.7 * (780.0 - w) / (780.0 - 700.0)

    gamma = 0.8

    def conv(c: float) -> int:
        if c <= 0.0:
            return 0
        return int(round(255.0 * ((c * factor) ** gamma)))

    return (conv(r), conv(g), conv(b))


def kelvin_to_rgb(temp_k: float) -> Tuple[int, int, int]:
    """
    Approximation classique de la couleur d'un corps noir en Kelvin.
    Intervalle utile ~ 1000K–40000K.
    """
    t = max(1000.0, min(40000.0, float(temp_k))) / 100.0

    # Rouge
    if t <= 66.0:
        r = 255
    else:
        r = 329.698727446 * ((t - 60.0) ** -0.1332047592)
        r = max(0, min(255, int(round(r))))

    # Vert
    if t <= 66.0:
        g = 99.4708025861 * math.log(t) - 161.1195681661
    else:
        g = 288.1221695283 * ((t - 60.0) ** -0.0755148492)
    g = max(0, min(255, int(round(g))))

    # Bleu
    if t >= 66.0:
        b = 255
    elif t <= 19.0:
        b = 0
    else:
        b = 138.5177312231 * math.log(t - 10.0) - 305.0447927307
        b = max(0, min(255, int(round(b))))

    return (r, g, b)

# ---------- 1) Modèle de données : engrenages & paths ----------

GEAR_TYPES = [
    "anneau",    # ring
    "roue",      # wheel
    "triangle",
    "carré",
    "barre",
    "croix",
    "oeil",
    "dsl",
    "modulaire",  # piste modulaire virtuelle (uniquement engrenage 1)
]

RELATIONS = [
    "stationnaire",  # pour le premier (au centre)
    "dedans",        # gear inside (hypotrochoïde)
    "dehors",        # gear outside (épitrochoïde)
]

@dataclass
class GearConfig:
    name: str = "Engrenage"
    gear_type: str = "anneau"   # anneau, roue, triangle, carré, barre, croix, oeil, modulaire
    size: int = 96              # taille de la roue / taille intérieure de l'anneau
    outer_size: int = 144       # anneau : taille extérieure / anneau modulaire
    relation: str = "stationnaire"  # stationnaire / dedans / dehors
    modular_notation: Optional[str] = None  # notation de piste si gear_type == "modulaire"
    dsl_expression: Optional[str] = None  # expression DSL si gear_type == "dsl"


@dataclass
class PathConfig:
    name: str = "Tracé"
    enable: bool = True
    hole_offset: float = 1.0
    phase_offset: float = 0.0
    color: str = "blue"            # chaîne telle que saisie / affichée
    color_norm: Optional[str] = None  # valeur normalisée (#rrggbb) pour le dessin
    stroke_width: float = 1.2
    zoom: float = 1.0
    translate_x: float = 0.0
    translate_y: float = 0.0
    rotate_deg: float = 0.0


@dataclass
class LayerConfig:
    name: str = "Couche"
    enable: bool = True
    zoom: float = 1.0                         # zoom de la couche
    translate_x: float = 0.0
    translate_y: float = 0.0
    rotate_deg: float = 0.0
    gears: List[GearConfig] = field(default_factory=list)  # 2 ou 3 engrenages
    paths: List[PathConfig] = field(default_factory=list)


# ---------- 2) GÉOMÉTRIE ----------

def radius_from_size(size: int) -> float:
    """
    Calcule le rayon d’un cercle de pas ayant une taille donnée.
    """
    if size <= 0:
        return 0.0
    return float(size) / (2.0 * math.pi)


def contact_size_for_relation(gear: GearConfig, relation: str) -> int:
    """
    Taille utilisée pour le contact, selon la relation.
    - anneau + 'dedans' : on utilise la taille intérieure (gear.size)
    - anneau + 'dehors' : on utilise la taille extérieure (gear.outer_size)
    - roue / autres : gear.size
    """
    if gear.gear_type == "anneau":
        if relation == "dehors":
            return gear.outer_size or gear.size
        else:
            return gear.size
    return gear.size


def phase_offset_turns(offset: float, size: int) -> float:
    """
    Convertit un décalage en unités (O) en fraction de tour (O/S).
    """
    if size <= 0:
        return 0.0
    return float(offset) / float(size)


def contact_radius_for_relation(gear: GearConfig, relation: str) -> float:
    """
    Rayon de contact pour un engrenage donné, selon la relation.
    """
    size = contact_size_for_relation(gear, relation)
    return radius_from_size(size)


def _curve_from_analytic_spec(spec, relation: str) -> BaseCurve:
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


def _curve_from_gear(gear: GearConfig, relation: str) -> Optional[BaseCurve]:
    if gear.gear_type == "modulaire" and gear.modular_notation:
        track = modular_tracks.build_track_from_notation(
            gear.modular_notation,
            inner_size=gear.size or 1,
            outer_size=gear.outer_size or (gear.size + 1),
            steps_per_unit=3,
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

    if gear.gear_type == "dsl" and gear.dsl_expression:
        spec = parse_analytic_expression(gear.dsl_expression)
        return _curve_from_analytic_spec(spec, relation)

    if gear.gear_type == "anneau":
        return build_circle(contact_size_for_relation(gear, relation))

    return build_circle(gear.size or 1)


def _gear_perimeter(gear: GearConfig, relation: str) -> float:
    if gear.gear_type == "dsl" and gear.dsl_expression:
        try:
            spec = parse_analytic_expression(gear.dsl_expression)
        except DslParseError:
            return float(contact_size_for_relation(gear, relation))
        curve = _curve_from_analytic_spec(spec, relation)
        return curve.length
    return float(contact_size_for_relation(gear, relation))


def generate_simple_circle_for_index(
    hole_offset: float,
    steps: int,
):
    """
    Fallback si on n’a pas assez d’engrenages : on simule un cercle
    dont le rayon dépend du trou indexé.
    On prend un rayon "référence" R_tip = 50 mm.
    Trou 0 : rayon = R_tip
    Trou n : R = R_tip - n
    R est clampé à >= 0.
    """
    R_tip = 50.0
    d = R_tip - hole_offset
    if d < 0:
        d = 0.0

    pts = []
    for i in range(steps):
        t = 2.0 * math.pi * i / (steps - 1)
        x = d * math.cos(t)
        y = d * math.sin(t)
        pts.append((x, y))
    return pts


def generate_trochoid_points_for_layer_path(
    layer: LayerConfig,
    path: PathConfig,
    steps: int = 5000,
):
    """
    Génère la courbe pour un path donné, en utilisant la configuration
    du layer (engrenages + organisation).

    Convention :
      - Le PREMIER engrenage de la couche (gears[0]) est stationnaire
        et centré en (0, 0).
      - Le DEUXIÈME engrenage (gears[1]) est mobile et porte les trous du path.
      - path.hole_offset est un float, peut être négatif.

    Si le premier engrenage est de type "modulaire", il représente une
    piste virtuelle SuperSpirograph, définie par :
      - g0.size        => taille intérieure de l’anneau de base
      - g0.outer_size  => taille extérieure de l’anneau de base
      - g0.modular_notation => notation de pièce (ex: "+A60+L144-E*+A72")
    La courbe est ensuite utilisée comme piste de contact pour le roulage.
    """

    hole_offset = float(path.hole_offset)

    # Pas assez d’engrenages : cercle simple + rotation
    if len(layer.gears) < 2:
        base_points = generate_simple_circle_for_index(hole_offset, steps)
        phase_turns = phase_offset_turns(path.phase_offset, 1)
        total_angle = math.pi / 2.0 - (2.0 * math.pi * phase_turns)

        cos_a = math.cos(total_angle)
        sin_a = math.sin(total_angle)
        rotated = []
        for (x, y) in base_points:
            xr = x * cos_a - y * sin_a
            yr = x * sin_a + y * cos_a
            rotated.append((xr, yr))
        return rotated

    g0 = layer.gears[0]  # stationnaire, au centre
    g1 = layer.gears[1]  # mobile, porte les trous

    relation = g1.relation

    try:
        base_curve = _curve_from_gear(g0, relation)
    except DslParseError:
        base_curve = None

    if base_curve is None or base_curve.length <= 0:
        base_points = generate_simple_circle_for_index(hole_offset, steps)
        phase_turns = phase_offset_turns(path.phase_offset, 1)
        total_angle = math.pi / 2.0 - (2.0 * math.pi * phase_turns)

        cos_a = math.cos(total_angle)
        sin_a = math.sin(total_angle)
        rotated = []
        for (x, y) in base_points:
            xr = x * cos_a - y * sin_a
            yr = x * sin_a + y * cos_a
            rotated.append((xr, yr))
        return rotated

    wheel_size = max(1.0, _gear_perimeter(g1, relation))
    r = radius_from_size(wheel_size)
    if g1.gear_type == "anneau":
        tip_size = g1.outer_size or g1.size
    elif g1.gear_type == "dsl" and g1.dsl_expression:
        tip_size = wheel_size
    else:
        tip_size = g1.size

    d = max(0.0, radius_from_size(tip_size) - hole_offset)

    if base_curve.closed:
        g = math.gcd(int(round(base_curve.length)), int(round(wheel_size))) or 1
        s_max = base_curve.length * (wheel_size / g)
    else:
        s_max = base_curve.length
    s_max = max(s_max, base_curve.length)

    side = 1 if relation == "dedans" else -1
    epsilon = side
    if relation == "dedans" and isinstance(base_curve, CircleCurve):
        alpha0 = math.pi
    else:
        alpha0 = 0.0

    base_points = []
    for i in range(steps):
        s = s_max * i / (steps - 1)
        x, y = pen_position(s, base_curve, r, d, side, alpha0, epsilon)
        base_points.append((x, y))

    phase_turns = phase_offset_turns(path.phase_offset, max(1, int(round(base_curve.length))))
    total_angle = math.pi / 2.0 - (2.0 * math.pi * phase_turns)

    cos_a = math.cos(total_angle)
    sin_a = math.sin(total_angle)
    rotated_points = []
    for (x, y) in base_points:
        xr = x * cos_a - y * sin_a
        yr = x * sin_a + y * cos_a
        rotated_points.append((xr, yr))

    return rotated_points


# ---------- 3) Validation de couleur ----------

def normalize_color_name(name: str) -> str:
    import re
    return re.sub(r"\s+", "", name.strip().lower())

def is_valid_color_name(name: str) -> bool:
    return normalize_color_name(name) in COLOR_NAME_TO_HEX

def resolve_color_to_hex(name: str) -> str:
    key = normalize_color_name(name)
    return COLOR_NAME_TO_HEX[key]  # KeyError if unknown

def normalize_color_string(s: str) -> Optional[str]:
    """
    Renvoie une version normalisée de la couleur en hex :
      - #rrggbb ou #rrggbbaa
    ou None si la couleur est invalide.

    Règles :
      - Si ça commence par # -> valeur hex, validée par regex.
      - Si c'est du type (H, S, L) -> HSL, H en degrés, S et L dans [0, 1].
      - Sinon -> nom de couleur passé par COLOR_NAME_TO_HEX.
    """
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None

    # 1) Hex direct (#rgb, #rgba, #rrggbb, #rrggbbaa)
    if s.startswith("#"):
        m = re.fullmatch(r"#([0-9a-fA-F]{3}|[0-9a-fA-F]{4}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})", s)
        if not m:
            return None
        return s.lower()

    # 2) HSL : (Hue, Saturation, Luminance)
    # Exemple : (120, 0.5, 0.4)
    m = re.fullmatch(
        r"\(\s*([+-]?\d*\.?\d+)\s*,\s*([+-]?\d*\.?\d+)\s*,\s*([+-]?\d*\.?\d+)\s*\)",
        s,
    )
    if m:
        try:
            h = float(m.group(1))
            sat = float(m.group(2))
            lum = float(m.group(3))

            # Hue en degrés -> [0, 360)
            h = h % 360.0
            # Saturation et luminance clampées à [0, 1]
            sat = max(0.0, min(1.0, sat))
            lum = max(0.0, min(1.0, lum))

            # colorsys.hls_to_rgb attend (h, l, s) avec h dans [0,1]
            r_f, g_f, b_f = colorsys.hls_to_rgb(h / 360.0, lum, sat)
            r = int(round(r_f * 255))
            g = int(round(g_f * 255))
            b = int(round(b_f * 255))
            return f"#{r:02x}{g:02x}{b:02x}"
        except ValueError:
            return None

    # 3) Nom de couleur via ton dictionnaire
    try:
        hexv = resolve_color_to_hex(s)
        return hexv.lower()
    except KeyError:
        return None
        return None


def is_valid_color_string(s: str) -> bool:
    """Couleur valide si normalize_color_string(s) ne renvoie pas None."""
    return normalize_color_string(s) is not None

class ColorSquare(QWidget):
    """
    Carré Hue / Saturation (H/S) avec Value (V) gérée par un slider externe.

    - Horizontal : Hue  [0..1]  (0 = 0°, 1 = 360°)
    - Vertical   : Saturation [0..1] (1 en haut, 0 en bas)

    Le signal colorChanged émet (h, s) dans [0,1] quand on clique/drag.
    """

    colorChanged = Signal(float, float)  # (h, s)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._h = 0.0  # 0..1
        self._s = 1.0  # 0..1
        self._v = 0.5  # 0..1 (géré par un slider externe, mais stocké ici pour dessiner)

    def set_hsv(self, h: float, s: float, v: float):
        self._h = max(0.0, min(1.0, h))
        self._s = max(0.0, min(1.0, s))
        self._v = max(0.0, min(1.0, v))
        self.update()

    def get_hsv(self):
        return self._h, self._s, self._v

    def _update_from_pos(self, pt: QPoint):
        w = max(1, self.width())
        h = max(1, self.height())
        x = min(max(pt.x(), 0), w - 1)
        y = min(max(pt.y(), 0), h - 1)

        # H = horizontal, S = vertical (1 en haut -> 0 en bas)
        self._h = x / (w - 1) if w > 1 else 0.0
        self._s = 1.0 - (y / (h - 1) if h > 1 else 0.0)

        self.colorChanged.emit(self._h, self._s)
        self.update()

    def mousePressEvent(self, event):
        pos = event.position().toPoint() if hasattr(event, "position") else event.pos()
        self._update_from_pos(pos)

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            pos = event.position().toPoint() if hasattr(event, "position") else event.pos()
            self._update_from_pos(pos)

class ColorPickerDialog(QDialog):
    """
    Sélecteur de couleur évolué :
      - Carré H/S + slider de Value (V)
      - Champ texte (nom, #hex, (H, S, L))
      - Liste filtrable de noms (COLOR_NAME_TO_HEX)
      - Harmonies : complémentaire, analogues, triadique, tétradique, tints & shades
      - Sélection par longueur d'onde (nm) et température (K)
    """

    def __init__(self, initial_text: str = "", lang: str = "fr", parent=None):
        super().__init__(parent)
        self.lang = localisation.normalize_language(lang) or "fr"
        self.setWindowTitle(tr(self.lang, "color_picker_title"))
        self.resize(820, 460)

        self._updating = False
        self._h = 0.0      # 0..1
        self._s = 1.0      # 0..1
        self._v = 0.5      # 0..1 (50 % par défaut)
        self._hex = "#ffffff"
        self._scheme_colors: List[str] = []

        main_layout = QHBoxLayout(self)

        # --- zone gauche : carré + slider V + prévisualisation + champs numériques ---
        left_layout = QVBoxLayout()

        # carré H/S + slider de Value
        sv_layout = QHBoxLayout()
        self.square = ColorSquare()
        self.value_slider = QSlider(Qt.Vertical)
        self.value_slider.setRange(0, 100)     # 0..100 -> V 0..1
        self.value_slider.setValue(50)         # V = 50%

        sv_layout.addWidget(self.square, 1)
        sv_layout.addWidget(self.value_slider)

        left_layout.addLayout(sv_layout)

        # prévisualisation + champ texte
        prev_layout = QHBoxLayout()
        self.preview_label = QLabel()
        self.preview_label.setFixedSize(60, 30)
        self.preview_label.setAutoFillBackground(True)

        self.text_edit = QLineEdit()
        self.text_edit.setPlaceholderText(tr(self.lang, "color_picker_text_placeholder"))
        prev_layout.addWidget(self.preview_label)
        prev_layout.addWidget(self.text_edit, 1)

        left_layout.addLayout(prev_layout)

        # ligne RGB + Wavelength + Kelvin
        numeric_layout = QHBoxLayout()

        def make_spin(label_txt: str):
            box = QVBoxLayout()
            lab = QLabel(label_txt)
            spin = QSpinBox()
            box.addWidget(lab)
            box.addWidget(spin)
            return box, spin

        rgb_r_box, self.r_spin = make_spin("R")
        rgb_g_box, self.g_spin = make_spin("G")
        rgb_b_box, self.b_spin = make_spin("B")
        self.r_spin.setRange(0, 255)
        self.g_spin.setRange(0, 255)
        self.b_spin.setRange(0, 255)

        # Wavelength (nm)
        wave_box = QVBoxLayout()
        wave_label = QLabel("λ (nm)")
        self.wave_spin = QDoubleSpinBox()
        self.wave_spin.setRange(380.0, 780.0)
        self.wave_spin.setDecimals(1)
        self.wave_spin.setSingleStep(1.0)
        wave_box.addWidget(wave_label)
        wave_box.addWidget(self.wave_spin)

        # Température (K)
        temp_box = QVBoxLayout()
        temp_label = QLabel("T (K)")
        self.temp_spin = QSpinBox()
        self.temp_spin.setRange(1000, 40000)
        self.temp_spin.setSingleStep(100)
        temp_box.addWidget(temp_label)
        temp_box.addWidget(self.temp_spin)

        numeric_layout.addLayout(rgb_r_box)
        numeric_layout.addLayout(rgb_g_box)
        numeric_layout.addLayout(rgb_b_box)
        numeric_layout.addSpacing(12)
        numeric_layout.addLayout(wave_box)
        numeric_layout.addLayout(temp_box)

        left_layout.addLayout(numeric_layout)

        # --- Harmonies : combo + 5 pastilles cliquables ---
        harmony_layout = QVBoxLayout()

        scheme_row = QHBoxLayout()
        scheme_label = QLabel(tr(self.lang, "color_picker_harmony_label"))
        self.scheme_combo = QComboBox()
        scheme_options = [
            ("none", tr(self.lang, "color_picker_scheme_none")),
            ("complementary", tr(self.lang, "color_picker_scheme_complementary")),
            ("analogous", tr(self.lang, "color_picker_scheme_analogous")),
            ("triadic", tr(self.lang, "color_picker_scheme_triadic")),
            ("tetradic", tr(self.lang, "color_picker_scheme_tetradic")),
            ("tints_shades", tr(self.lang, "color_picker_scheme_tints_shades")),
        ]
        for key, label in scheme_options:
            self.scheme_combo.addItem(label, key)
        scheme_row.addWidget(scheme_label)
        scheme_row.addWidget(self.scheme_combo, 1)

        harmony_layout.addLayout(scheme_row)

        self.scheme_buttons: List[QPushButton] = []
        btn_row = QHBoxLayout()
        for _ in range(5):
            b = QPushButton()
            b.setFixedSize(32, 32)
            b.setFlat(True)
            b.clicked.connect(self.on_scheme_button_clicked)
            self.scheme_buttons.append(b)
            btn_row.addWidget(b)
        harmony_layout.addLayout(btn_row)

        left_layout.addLayout(harmony_layout)

        main_layout.addLayout(left_layout, 2)

        # --- zone droite : liste de noms de couleurs ---
        right_layout = QVBoxLayout()
        search_layout = QHBoxLayout()
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText(tr(self.lang, "color_picker_search_placeholder"))
        btn_clear = QPushButton(tr(self.lang, "color_picker_clear"))
        btn_clear.clicked.connect(self.search_edit.clear)
        search_layout.addWidget(self.search_edit, 1)
        search_layout.addWidget(btn_clear)

        right_layout.addLayout(search_layout)

        self.list_widget = QListWidget()
        right_layout.addWidget(self.list_widget, 1)

        main_layout.addLayout(right_layout, 1)

        # --- boutons OK / Annuler ---
        btn_layout = QHBoxLayout()
        btn_ok = QPushButton(tr(self.lang, "dlg_ok"))
        btn_cancel = QPushButton(tr(self.lang, "dlg_cancel"))
        btn_ok.clicked.connect(self.accept)
        btn_cancel.clicked.connect(self.reject)
        btn_layout.addStretch(1)
        btn_layout.addWidget(btn_ok)
        btn_layout.addWidget(btn_cancel)

        main_layout.addLayout(btn_layout)

        # Connexions
        self.value_slider.valueChanged.connect(self.on_value_changed)
        self.square.colorChanged.connect(self.on_hs_changed)
        self.text_edit.editingFinished.connect(self.on_text_edited)
        self.r_spin.valueChanged.connect(self.on_rgb_spin_changed)
        self.g_spin.valueChanged.connect(self.on_rgb_spin_changed)
        self.b_spin.valueChanged.connect(self.on_rgb_spin_changed)
        self.wave_spin.valueChanged.connect(self.on_wave_changed)
        self.temp_spin.valueChanged.connect(self.on_temp_changed)
        self.search_edit.textChanged.connect(self.on_search_changed)
        self.list_widget.itemDoubleClicked.connect(self.on_list_double_clicked)
        self.scheme_combo.currentIndexChanged.connect(self.update_scheme_palette)

        self.populate_color_list()
        self.set_from_text(initial_text or "#ffffff")

    # -------- utilitaires liste de couleurs --------

    def populate_color_list(self):
        self.list_widget.clear()
        from PySide6.QtGui import QPixmap

        for name in sorted(COLOR_NAME_TO_HEX.keys()):
            hexv = COLOR_NAME_TO_HEX[name]
            item = QListWidgetItem(name)
            item.setData(Qt.UserRole, hexv)
            pix = QImage(16, 16, QImage.Format_RGB32)
            pix.fill(QColor(hexv))
            item.setIcon(QPixmap.fromImage(pix))
            self.list_widget.addItem(item)

    def on_search_changed(self, text: str):
        t = text.strip().lower()
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            name = item.text().lower()
            item.setHidden(t not in name)

    def on_list_double_clicked(self, item: QListWidgetItem):
        if not item:
            return
        name = item.text()
        hexv = item.data(Qt.UserRole)
        self._set_from_hex_and_text(hexv, name)

    # -------- Harmonies --------

    def update_scheme_palette(self, _index: int = 0):
        scheme = self.scheme_combo.currentData()

        c = QColor()
        c.setHsvF(self._h, self._s, self._v)
        base_h, base_s, base_v, _ = c.getHsvF()

        colors: List[str] = []

        def add_h_offset(deg_offset: float, s_mult: float = 1.0, v_mult: float = 1.0):
            h = (base_h + deg_offset / 360.0) % 1.0
            s = max(0.0, min(1.0, base_s * s_mult))
            v = max(0.0, min(1.0, base_v * v_mult))
            cc = QColor()
            cc.setHsvF(h, s, v)
            colors.append(f"#{cc.red():02x}{cc.green():02x}{cc.blue():02x}")

        if scheme == "none":
            colors = []
        elif scheme == "complementary":
            add_h_offset(0)
            add_h_offset(180)
        elif scheme == "analogous":
            add_h_offset(-30)
            add_h_offset(0)
            add_h_offset(30)
        elif scheme == "triadic":
            add_h_offset(0)
            add_h_offset(120)
            add_h_offset(240)
        elif scheme == "tetradic":
            add_h_offset(0)
            add_h_offset(90)
            add_h_offset(180)
            add_h_offset(270)
        elif scheme == "tints_shades":
            for v_mult in (1.2, 1.0, 0.8, 0.6, 0.4):
                v = max(0.0, min(1.0, base_v * v_mult))
                cc = QColor()
                cc.setHsvF(base_h, base_s, v)
                colors.append(f"#{cc.red():02x}{cc.green():02x}{cc.blue():02x}")

        self._scheme_colors = colors[:5]
        for i, btn in enumerate(self.scheme_buttons):
            if i < len(self._scheme_colors):
                hexv = self._scheme_colors[i]
                btn.setEnabled(True)
                btn.setStyleSheet(f"background-color: {hexv}; border: 1px solid #202020;")
            else:
                btn.setEnabled(False)
                btn.setStyleSheet("background-color: none; border: none;")

    def on_scheme_button_clicked(self):
        if self._updating:
            return
        btn = self.sender()
        if not isinstance(btn, QPushButton):
            return
        if btn not in self.scheme_buttons:
            return
        idx = self.scheme_buttons.index(btn)
        if idx >= len(self._scheme_colors):
            return
        hexv = self._scheme_colors[idx]
        self._set_from_hex_and_text(hexv, hexv)

    # -------- Réactions aux changements numériques / HSV --------

    def on_value_changed(self, value: int):
        if self._updating:
            return
        self._v = value / 100.0
        self._update_from_hsv()

    def on_hs_changed(self, h: float, s: float):
        if self._updating:
            return
        self._h = h
        self._s = s
        self._update_from_hsv()

    def on_rgb_spin_changed(self, _value: int):
        if self._updating:
            return
        r = self.r_spin.value()
        g = self.g_spin.value()
        b = self.b_spin.value()
        c = QColor(r, g, b)
        h, s, v, _ = c.getHsvF()
        self._h, self._s, self._v = h, s, v
        self._update_from_hsv()
        self.text_edit.setText(self._hex)

    def on_wave_changed(self, value: float):
        if self._updating:
            return
        r, g, b = wavelength_to_rgb(value)
        c = QColor(r, g, b)
        h, s, v, _ = c.getHsvF()
        self._h, self._s, self._v = h, s, v
        self._update_from_hsv()
        self.text_edit.setText(self._hex)

    def on_temp_changed(self, value: int):
        if self._updating:
            return
        r, g, b = kelvin_to_rgb(value)
        c = QColor(r, g, b)
        h, s, v, _ = c.getHsvF()
        self._h, self._s, self._v = h, s, v
        self._update_from_hsv()
        self.text_edit.setText(self._hex)

    def on_text_edited(self):
        if self._updating:
            return
        text = self.text_edit.text().strip()
        if not text:
            return
        hexv = normalize_color_string(text)
        if hexv is None:
            self.text_edit.setText(self._hex)
            return
        self._set_from_hex_and_text(hexv, text)

    def _set_from_hex_and_text(self, hexv: str, text_repr: str):
        c = QColor(hexv)
        h, s, v, _ = c.getHsvF()
        self._h, self._s, self._v = h, s, v
        self._hex = hexv

        self._updating = True
        try:
            self.square.set_hsv(self._h, self._s, self._v)
            self.value_slider.setValue(int(round(self._v * 100)))
            self._update_widgets_from_color(c, keep_text=False)
            self.text_edit.setText(text_repr)
            self._update_preview()
            self.update_scheme_palette()
        finally:
            self._updating = False

    def _update_from_hsv(self):
        c = QColor()
        c.setHsvF(self._h, self._s, self._v)
        self._hex = f"#{c.red():02x}{c.green():02x}{c.blue():02x}"

        self._updating = True
        try:
            self.square.set_hsv(self._h, self._s, self._v)
            self.value_slider.setValue(int(round(self._v * 100)))
            self._update_widgets_from_color(c, keep_text=False)
            current = self.text_edit.text().strip()
            if not current or normalize_color_string(current) != self._hex:
                self.text_edit.setText(self._hex)
            self._update_preview()
            self.update_scheme_palette()
        finally:
            self._updating = False

    def _update_widgets_from_color(self, c: QColor, keep_text: bool):
        self._hex = f"#{c.red():02x}{c.green():02x}{c.blue():02x}"
        self.r_spin.setValue(c.red())
        self.g_spin.setValue(c.green())
        self.b_spin.setValue(c.blue())
        if not keep_text:
            pass
        self._update_preview()

    def _update_preview(self):
        pal = self.preview_label.palette()
        pal.setColor(self.preview_label.backgroundRole(), QColor(self._hex))
        self.preview_label.setPalette(pal)

    # -------- API publique --------

    def set_from_text(self, text: str):
        hexv = normalize_color_string(text)
        if hexv is None:
            hexv = "#ffffff"
            text = "#ffffff"
        self._set_from_hex_and_text(hexv, text)

    def result_text(self) -> str:
        txt = self.text_edit.text().strip()
        if txt and normalize_color_string(txt) is not None:
            return txt
        return self._hex

    @staticmethod
    def get_color(initial_text: str = "", lang: str = "fr", parent=None) -> Optional[str]:
        dlg = ColorPickerDialog(initial_text=initial_text, lang=lang, parent=parent)
        if dlg.exec() == QDialog.Accepted:
            return dlg.result_text()
        return None

# ---------- 4) SVG : plusieurs layers & paths ----------

def layers_to_svg(
    layers: List[LayerConfig],
    width: int = 1000,
    height: int = 1000,
    bg_color: str = "#ffffff",
    points_per_path: int = 6000,
    show_tracks: bool = True,
    return_render_data: bool = False,
) -> str:
    """
    Convertit une liste de LayerConfig -> SVG string.
    Chaque layer activé devient un <g>, chaque path un <path>.
    On applique le zoom de la couche et du tracé avant le centrage/scaling global.
    Quand return_render_data=True, renvoie aussi une structure réutilisable pour
    l'animation (points déjà transformés en pixels).
    """
    def apply_transform(points, rotate_deg: float, translate_x: float, translate_y: float):
        if not points:
            return points
        if (
            abs(rotate_deg) < 1e-9
            and abs(translate_x) < 1e-9
            and abs(translate_y) < 1e-9
        ):
            return points
        angle = math.radians(rotate_deg)
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        return [
            (
                (x * cos_a - y * sin_a) + translate_x,
                (x * sin_a + y * cos_a) + translate_y,
            )
            for (x, y) in points
        ]
    all_points = []
    trace_points = []
    rendered_paths = []  # (layer_name, layer_zoom, path_config, points, path_zoom)
    render_paths = []
    render_tracks = []

    for layer in layers:
        if not layer.enable:
            continue
        layer_zoom = getattr(layer, "zoom", 1.0)
        layer_rotate = getattr(layer, "rotate_deg", 0.0)
        layer_tx = getattr(layer, "translate_x", 0.0)
        layer_ty = getattr(layer, "translate_y", 0.0)

        layer_track_points = None
        layer_track_width_mm = None
        if show_tracks:
            if (
                layer.gears
                and layer.gears[0].gear_type == "modulaire"
                and getattr(layer.gears[0], "modular_notation", "")
            ):
                g0 = layer.gears[0]
                relation = "dedans"
                wheel_size_rel = 1
                if len(layer.gears) > 1:
                    g1_tmp = layer.gears[1]
                    relation = getattr(g1_tmp, "relation", "dedans") or "dedans"
                    wheel_size_rel = max(1, contact_size_for_relation(g1_tmp, relation))

                inner_size = max(1, int(g0.size))
                outer_size = int(g0.outer_size) if g0.outer_size else inner_size
                outer_size = max(outer_size, inner_size)

                track, bundle = modular_tracks.build_track_and_bundle_from_notation(
                    notation=g0.modular_notation,
                    wheel_size=wheel_size_rel,
                    hole_offset=0.0,
                    steps=2,
                    relation=relation,
                    phase_offset=0.0,
                    inner_size=inner_size,
                    outer_size=outer_size,
                )
                if track.segments:
                    centerline, _, _, half_w = modular_tracks.compute_track_polylines(
                        track, samples=800, half_width=bundle.context.half_width
                    )
                    layer_track_points = centerline
                    layer_track_width_mm = (half_w * 2.0) * layer_zoom
        layer_paths = []
        for path in layer.paths:
            if not path.enable:
                continue
            pts = generate_trochoid_points_for_layer_path(
                layer,
                path,
                steps=points_per_path,
            )
            if not pts:
                continue
            path_zoom = getattr(path, "zoom", 1.0)
            zoom = layer_zoom * path_zoom
            pts_zoomed = [(x * zoom, y * zoom) for (x, y) in pts]
            if pts_zoomed:
                path_cx = sum(p[0] for p in pts_zoomed) / len(pts_zoomed)
                path_cy = sum(p[1] for p in pts_zoomed) / len(pts_zoomed)
            else:
                path_cx = 0.0
                path_cy = 0.0
            shifted_points = [(x - path_cx, y - path_cy) for (x, y) in pts_zoomed]
            path_rotate = getattr(path, "rotate_deg", 0.0)
            path_tx = getattr(path, "translate_x", 0.0)
            path_ty = getattr(path, "translate_y", 0.0)
            path_transformed = apply_transform(
                shifted_points, path_rotate, path_tx, path_ty
            )
            layer_transformed = apply_transform(
                path_transformed, layer_rotate, layer_tx, layer_ty
            )
            layer_paths.append((path, layer_transformed, path_zoom))

        if layer_paths:
            for path_cfg, shifted_points, path_zoom in layer_paths:
                rendered_paths.append(
                    (layer.name, layer_zoom, path_cfg, shifted_points, path_zoom)
                )
                all_points.extend(shifted_points)
                trace_points.extend(shifted_points)

        if show_tracks and layer_track_points:
            track_zoomed = [
                (x * layer_zoom, y * layer_zoom) for (x, y) in layer_track_points
            ]
            if track_zoomed:
                track_cx = sum(p[0] for p in track_zoomed) / len(track_zoomed)
                track_cy = sum(p[1] for p in track_zoomed) / len(track_zoomed)
            else:
                track_cx = 0.0
                track_cy = 0.0
            shifted_track = [(x - track_cx, y - track_cy) for (x, y) in track_zoomed]
            transformed_track = apply_transform(
                shifted_track, layer_rotate, layer_tx, layer_ty
            )
            render_tracks.append(
                {
                    "layer_name": layer.name,
                    "points": transformed_track,
                    "stroke_width_mm": layer_track_width_mm,
                }
            )
            all_points.extend(transformed_track)

    if not all_points:
        svg_empty = f'''<?xml version="1.0" standalone="no"?>
<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}"
     xmlns="http://www.w3.org/2000/svg" version="1.1">
  <rect x="0" y="0" width="{width}" height="{height}" fill="{bg_color}"/>
</svg>
'''
        if return_render_data:
            return svg_empty, None
        return svg_empty

    xs = [p[0] for p in all_points]
    ys = [p[1] for p in all_points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    dx = max_x - min_x
    dy = max_y - min_y
    if dx == 0:
        dx = 1
    if dy == 0:
        dy = 1

    scale = 0.8 * min(width / dx, height / dy)
    centroid_points = trace_points or all_points
    total_points = len(centroid_points)
    cx = sum(p[0] for p in centroid_points) / total_points
    cy = sum(p[1] for p in centroid_points) / total_points

    def transform(p):
        x, y = p
        x = (x - cx) * scale + width / 2.0
        y = (cy - y) * scale + height / 2.0  # inverser Y pour repère math
        return x, y

    svg_parts = [
        '<?xml version="1.0" standalone="no"?>',
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}"',
        '     xmlns="http://www.w3.org/2000/svg" version="1.1">',
        f'  <rect x="0" y="0" width="{width}" height="{height}" fill="{bg_color}"/>',
    ]

    tracks_out = []

    for layer in layers:
        if not layer.enable:
            continue
        layer_paths = [rp for rp in rendered_paths if rp[0] == layer.name]
        layer_tracks = [rt for rt in render_tracks if rt["layer_name"] == layer.name]
        if not layer_paths and not layer_tracks:
            continue

        svg_parts.append(f'  <g id="layer-{layer.name}">')
        for track_entry in layer_tracks:
            t_points = [transform(p) for p in track_entry["points"]]
            stroke_px = max(
                1.0,
                (track_entry.get("stroke_width_mm") or 0.0) * scale,
            )
            if len(t_points) >= 2:
                x0, y0 = t_points[0]
                t_cmds = [f"M {x0:.3f} {y0:.3f}"]
                for (x, y) in t_points[1:]:
                    t_cmds.append(f"L {x:.3f} {y:.3f}")
                tracks_out.append(
                    {"points": t_points, "stroke_width": stroke_px}
                )

        for _, _, path_cfg, pts_zoomed, _ in layer_paths:
            t_points = [transform(p) for p in pts_zoomed]
            x0, y0 = t_points[0]
            path_cmds = [f"M {x0:.3f} {y0:.3f}"]
            for (x, y) in t_points[1:]:
                path_cmds.append(f"L {x:.3f} {y:.3f}")
            path_d = " ".join(path_cmds)

            stroke_color = getattr(path_cfg, "color_norm", None)
            if not stroke_color:
                # sécurité : normalisation à la volée si pas encore calculée (vieux JSON, etc.)
                stroke_color = normalize_color_string(path_cfg.color) or "#000000"
                path_cfg.color_norm = stroke_color

            render_paths.append(
                {
                    "layer_name": layer.name,
                    "path_name": path_cfg.name,
                    "points": t_points,
                    "color": stroke_color,
                    "stroke_width": path_cfg.stroke_width,
                }
            )

            svg_parts.append(
                f'    <path d="{path_d}" fill="none" '
                f'stroke="{stroke_color}" stroke-width="{path_cfg.stroke_width}" '
                f'id="{layer.name}-{path_cfg.name}"/>'
            )
        svg_parts.append('  </g>')

    svg_parts.append('</svg>')
    svg_result = "\n".join(svg_parts)
    if return_render_data:
        render_data = {
            "width": width,
            "height": height,
            "bg_color": bg_color,
            "paths": render_paths,
            "tracks": tracks_out,
        }
        return svg_result, render_data
    return svg_result


# ---------- 5) Dialogues d’édition ----------

class LayerEditDialog(QDialog):
    """
    Édition d’un layer :
      - nom
      - zoom
      - translation / rotation
      - 2 ou 3 engrenages (type, tailles, relation)
      - pour un anneau : tailles extérieures / intérieures
    """

    def __init__(
        self,
        layer: LayerConfig,
        lang: str = "fr",
        parent=None,
    ):
        super().__init__(parent)
        self.lang = lang
        self.setWindowTitle(tr(self.lang, "dlg_layer_edit_title"))
        self.layer = layer

        layout = QFormLayout(self)

        self.name_edit = QLineEdit(self.layer.name)
        self.zoom_spin = QDoubleSpinBox()
        self.zoom_spin.setRange(0.01, 100.0)
        self.zoom_spin.setDecimals(3)
        self.zoom_spin.setValue(getattr(self.layer, "zoom", 1.0))

        self.translate_x_spin = QDoubleSpinBox()
        self.translate_x_spin.setRange(-10000.0, 10000.0)
        self.translate_x_spin.setDecimals(3)
        self.translate_x_spin.setValue(getattr(self.layer, "translate_x", 0.0))

        self.translate_y_spin = QDoubleSpinBox()
        self.translate_y_spin.setRange(-10000.0, 10000.0)
        self.translate_y_spin.setDecimals(3)
        self.translate_y_spin.setValue(getattr(self.layer, "translate_y", 0.0))

        self.rotate_spin = QDoubleSpinBox()
        self.rotate_spin.setRange(-360.0, 360.0)
        self.rotate_spin.setDecimals(3)
        self.rotate_spin.setValue(getattr(self.layer, "rotate_deg", 0.0))

        self.num_gears_spin = QSpinBox()
        self.num_gears_spin.setRange(2, 3)
        current_gears = max(2, min(3, len(self.layer.gears)))
        self.num_gears_spin.setValue(current_gears)

        layout.addRow(tr(self.lang, "dlg_layer_name"), self.name_edit)
        layout.addRow(tr(self.lang, "dlg_layer_zoom"), self.zoom_spin)
        layout.addRow(tr(self.lang, "dlg_layer_translate_x"), self.translate_x_spin)
        layout.addRow(tr(self.lang, "dlg_layer_translate_y"), self.translate_y_spin)
        layout.addRow(tr(self.lang, "dlg_layer_rotate"), self.rotate_spin)
        layout.addRow(tr(self.lang, "dlg_layer_num_gears"), self.num_gears_spin)

        self.gear_widgets = []
        for i in range(3):
            group_label = QLabel(tr(self.lang, "dlg_layer_gear_label").format(index=i + 1))
            gear_name_edit = QLineEdit()
            gear_type_combo = QComboBox()
            for gear_type in GEAR_TYPES:
                gear_type_combo.addItem(gear_type_label(gear_type, self.lang), gear_type)
            size_spin = QSpinBox()
            size_spin.setRange(1, 10000)
            outer_spin = QSpinBox()
            outer_spin.setRange(0, 20000)
            rel_combo = QComboBox()
            for relation in RELATIONS:
                rel_combo.addItem(relation_label(relation, self.lang), relation)

            # Édition de la notation modulaire (visible seulement pour engrenage 1 + type "modulaire")
            modular_edit = QLineEdit()
            modular_label = QLabel(tr(self.lang, "dlg_layer_gear_modular_notation"))
            modular_button = QPushButton("…")
            modular_button.setFixedWidth(28)
            dsl_edit = QLineEdit()
            dsl_label = QLabel(tr(self.lang, "dlg_layer_gear_dsl_expression"))

            sub = QVBoxLayout()
            sub.addWidget(group_label)

            row1 = QHBoxLayout()
            row1.addWidget(QLabel(tr(self.lang, "dlg_layer_gear_name")))
            row1.addWidget(gear_name_edit)
            sub.addLayout(row1)

            row2 = QHBoxLayout()
            row2.addWidget(QLabel(tr(self.lang, "dlg_layer_gear_type")))
            row2.addWidget(gear_type_combo)
            sub.addLayout(row2)

            row3 = QHBoxLayout()
            label_size = QLabel(tr(self.lang, "dlg_layer_gear_size"))
            row3.addWidget(label_size)
            row3.addWidget(size_spin)
            sub.addLayout(row3)

            row4 = QHBoxLayout()
            label_outer = QLabel(tr(self.lang, "dlg_layer_gear_outer"))
            row4.addWidget(label_outer)
            row4.addWidget(outer_spin)
            sub.addLayout(row4)

            row5 = QHBoxLayout()
            row5.addWidget(QLabel(tr(self.lang, "dlg_layer_gear_relation")))
            row5.addWidget(rel_combo)
            sub.addLayout(row5)

            # Ligne pour la notation modulaire
            row6 = QHBoxLayout()
            row6.addWidget(modular_label)
            row6.addWidget(modular_edit)
            row6.addWidget(modular_button)
            sub.addLayout(row6)

            # Ligne pour l'expression DSL
            row7 = QHBoxLayout()
            row7.addWidget(dsl_label)
            row7.addWidget(dsl_edit)
            sub.addLayout(row7)

            layout.addRow(sub)

            # Restreindre "modulaire" aux engrenages d'indice 0 uniquement
            if i > 0:
                idx_mod = gear_type_combo.findData("modulaire")
                if idx_mod >= 0:
                    gear_type_combo.removeItem(idx_mod)

            gw = dict(
                index=i,
                name_edit=gear_name_edit,
                type_combo=gear_type_combo,
                size_spin=size_spin,
                outer_spin=outer_spin,
                outer_label=label_outer,
                rel_combo=rel_combo,
                modular_label=modular_label,
                modular_edit=modular_edit,
                modular_button=modular_button,
                dsl_label=dsl_label,
                dsl_edit=dsl_edit,
            )
            modular_button.clicked.connect(
                lambda _checked=False, gw=gw: self._open_modular_editor_from_widget(gw)
            )
            self.gear_widgets.append(gw)

            gear_type_combo.currentTextChanged.connect(
                lambda text, gw=gw: self._update_gear_widget_visibility(gw)
            )

        # Initialiser à partir du layer
        for idx, gw in enumerate(self.gear_widgets):
            if idx < len(self.layer.gears):
                g = self.layer.gears[idx]
            else:
                if idx == 0:
                    g = GearConfig(
                        name=tr(self.lang, "default_ring_name"),
                        gear_type="anneau",
                        size=105,
                        outer_size=150,
                        relation="stationnaire",
                    )
                else:
                    g = GearConfig(
                        name=tr(self.lang, "default_wheel_name"),
                        gear_type="roue",
                        size=30,
                        relation="dedans",
                    )

            gw["name_edit"].setText(g.name)
            # Si jamais un fichier ancien contient "modulaire" sur un engrenage > 0, on le ramène à "roue"
            gear_type = g.gear_type
            if idx > 0 and gear_type == "modulaire":
                gear_type = "roue"

            type_index = gw["type_combo"].findData(gear_type)
            if type_index < 0:
                type_index = 0
            gw["type_combo"].setCurrentIndex(type_index)

            gw["size_spin"].setValue(g.size)
            gw["outer_spin"].setValue(g.outer_size if g.outer_size > 0 else 0)
            gw["modular_edit"].setText(getattr(g, "modular_notation", "") or "")
            gw["dsl_edit"].setText(getattr(g, "dsl_expression", "") or "")

            rel_index = gw["rel_combo"].findData(g.relation)
            if rel_index < 0:
                rel_index = 0
            gw["rel_combo"].setCurrentIndex(rel_index)
            self._update_gear_widget_visibility(gw)

        # premier engrenage stationnaire
        self.gear_widgets[0]["rel_combo"].setCurrentIndex(0)
        self.gear_widgets[0]["rel_combo"].setEnabled(False)

        btn_box = QHBoxLayout()
        btn_ok = QPushButton(tr(self.lang, "dlg_ok"))
        btn_cancel = QPushButton(tr(self.lang, "dlg_cancel"))
        btn_ok.clicked.connect(self.accept)
        btn_cancel.clicked.connect(self.reject)
        btn_box.addWidget(btn_ok)
        btn_box.addWidget(btn_cancel)
        layout.addRow(btn_box)

    def _update_gear_widget_visibility(self, gw: dict):
        t = gw["type_combo"].currentData() or gw["type_combo"].currentText()
        idx = gw.get("index", 0)

        # "anneau" et "modulaire" ont des tailles intérieures / extérieures
        is_ring_like = (t == "anneau" or t == "modulaire")
        gw["outer_label"].setVisible(is_ring_like)
        gw["outer_spin"].setVisible(is_ring_like)

        # La notation modulaire n’est visible que pour l’engrenage 1 et le type "modulaire"
        is_modular = (t == "modulaire" and idx == 0)
        gw["modular_label"].setVisible(is_modular)
        gw["modular_edit"].setVisible(is_modular)
        gw["modular_button"].setVisible(is_modular)
        is_dsl = t == "dsl"
        gw["dsl_label"].setVisible(is_dsl)
        gw["dsl_edit"].setVisible(is_dsl)

    def _open_modular_editor_from_widget(self, gw: dict):
        if gw.get("index", 0) != 0:
            return
        if (gw["type_combo"].currentData() or gw["type_combo"].currentText()) != "modulaire":
            return

        notation = gw["modular_edit"].text().strip()
        inner_size = gw["size_spin"].value()
        outer_size = gw["outer_spin"].value() or inner_size

        dlg = ModularTrackEditorDialog(
            lang=self.lang,
            parent=self,
            initial_notation=notation,
            inner_size=inner_size,
            outer_size=outer_size,
        )
        if dlg.exec() == QDialog.Accepted:
            gw["modular_edit"].setText(dlg.result_notation())

    def accept(self):
        self.layer.name = self.name_edit.text().strip() or tr(self.lang, "default_layer_name")
        self.layer.zoom = self.zoom_spin.value()
        self.layer.translate_x = self.translate_x_spin.value()
        self.layer.translate_y = self.translate_y_spin.value()
        self.layer.rotate_deg = self.rotate_spin.value()
        num_gears = self.num_gears_spin.value()

        new_gears: List[GearConfig] = []
        for i in range(num_gears):
            gw = self.gear_widgets[i]
            name = gw["name_edit"].text().strip() or tr(self.lang, "default_gear_name").format(index=i + 1)
            gear_type = gw["type_combo"].currentData() or gw["type_combo"].currentText()

            # Sécurité : on n’autorise "modulaire" que pour le premier engrenage
            if i > 0 and gear_type == "modulaire":
                gear_type = "roue"

            size = gw["size_spin"].value()
            outer_size = gw["outer_spin"].value() if gear_type in ("anneau", "modulaire") else 0

            rel = gw["rel_combo"].currentData() or gw["rel_combo"].currentText()
            if i == 0:
                rel = "stationnaire"

            modular_notation = None
            if gear_type == "modulaire":
                txt = gw["modular_edit"].text().strip()
                if txt:
                    modular_notation = txt

            dsl_expression = None
            if gear_type == "dsl":
                txt = gw["dsl_edit"].text().strip()
                if txt:
                    dsl_expression = txt

            new_gears.append(
                GearConfig(
                    name=name,
                    gear_type=gear_type,
                    size=size,
                    outer_size=outer_size,
                    relation=rel,
                    modular_notation=modular_notation,
                    dsl_expression=dsl_expression,
                )
            )
        self.layer.gears = new_gears

        super().accept()


class PathEditDialog(QDialog):
    """
    Path :
      - hole_offset (float, positif ou négatif)
      - décalage de phase (déplacement en unités le long de la piste)
      - couleur (CSS4 ou hex) avec validation X11/CSS4/hex
      - largeur de trait
      - zoom
      - translation / rotation
    """

    def __init__(self, path: PathConfig, lang: str = "fr", parent=None):
        super().__init__(parent)
        self.lang = lang
        self.setWindowTitle(tr(self.lang, "dlg_path_edit_title"))
        self.path = path

        layout = QFormLayout(self)

        self.name_edit = QLineEdit(self.path.name)

        self.hole_spin = QDoubleSpinBox()
        self.hole_spin.setRange(-1000.0, 1000.0)
        self.hole_spin.setDecimals(3)
        self.hole_spin.setValue(self.path.hole_offset)

        self.phase_spin = QDoubleSpinBox()
        self.phase_spin.setRange(-1000.0, 1000.0)
        self.phase_spin.setDecimals(3)
        self.phase_spin.setValue(self.path.phase_offset)

        self.color_edit = QLineEdit(self.path.color)
        btn_pick = QPushButton("…")
        btn_pick.setFixedWidth(30)
        btn_pick.clicked.connect(self.open_color_picker)

        color_row = QHBoxLayout()
        color_row.addWidget(self.color_edit, 1)
        color_row.addWidget(btn_pick)

        self.stroke_spin = QDoubleSpinBox()
        self.stroke_spin.setRange(0.1, 50.0)
        self.stroke_spin.setDecimals(3)
        self.stroke_spin.setValue(self.path.stroke_width)

        self.zoom_spin = QDoubleSpinBox()
        self.zoom_spin.setRange(0.01, 100.0)
        self.zoom_spin.setDecimals(3)
        self.zoom_spin.setValue(getattr(self.path, "zoom", 1.0))

        self.translate_x_spin = QDoubleSpinBox()
        self.translate_x_spin.setRange(-10000.0, 10000.0)
        self.translate_x_spin.setDecimals(3)
        self.translate_x_spin.setValue(getattr(self.path, "translate_x", 0.0))

        self.translate_y_spin = QDoubleSpinBox()
        self.translate_y_spin.setRange(-10000.0, 10000.0)
        self.translate_y_spin.setDecimals(3)
        self.translate_y_spin.setValue(getattr(self.path, "translate_y", 0.0))

        self.rotate_spin = QDoubleSpinBox()
        self.rotate_spin.setRange(-360.0, 360.0)
        self.rotate_spin.setDecimals(3)
        self.rotate_spin.setValue(getattr(self.path, "rotate_deg", 0.0))

        layout.addRow(tr(self.lang, "dlg_path_name"), self.name_edit)
        layout.addRow(tr(self.lang, "dlg_path_hole_index"), self.hole_spin)
        layout.addRow(tr(self.lang, "dlg_path_phase"), self.phase_spin)
        layout.addRow(tr(self.lang, "dlg_path_color"), color_row)
        layout.addRow(tr(self.lang, "dlg_path_width"), self.stroke_spin)
        layout.addRow(tr(self.lang, "dlg_path_zoom"), self.zoom_spin)
        layout.addRow(tr(self.lang, "dlg_path_translate_x"), self.translate_x_spin)
        layout.addRow(tr(self.lang, "dlg_path_translate_y"), self.translate_y_spin)
        layout.addRow(tr(self.lang, "dlg_path_rotate"), self.rotate_spin)

        btn_box = QHBoxLayout()
        btn_ok = QPushButton(tr(self.lang, "dlg_ok"))
        btn_cancel = QPushButton(tr(self.lang, "dlg_cancel"))
        btn_ok.clicked.connect(self.accept)
        btn_cancel.clicked.connect(self.reject)
        btn_box.addWidget(btn_ok)
        btn_box.addWidget(btn_cancel)
        layout.addRow(btn_box)

    def open_color_picker(self):
        current = self.color_edit.text().strip()
        picked = ColorPickerDialog.get_color(initial_text=current, lang=self.lang, parent=self)
        if picked is not None:
            self.color_edit.setText(picked)

    def accept(self):
        self.path.name = self.name_edit.text().strip() or tr(self.lang, "default_path_name")
        self.path.hole_offset = self.hole_spin.value()
        self.path.phase_offset = self.phase_spin.value()

        new_color_input = self.color_edit.text().strip() or "#000000"
        norm_color = normalize_color_string(new_color_input)
        if norm_color is None:
            QMessageBox.warning(
                self,
                tr(self.lang, "color_invalid_title"),
                tr(self.lang, "color_invalid_text"),
            )
            return

        # On garde ce que l'utilisateur a saisi pour l'affichage / JSON…
        self.path.color = new_color_input
        # …et on stocke la forme normalisée pour le rendu.
        self.path.color_norm = norm_color
        self.path.stroke_width = self.stroke_spin.value()
        self.path.zoom = self.zoom_spin.value()
        self.path.translate_x = self.translate_x_spin.value()
        self.path.translate_y = self.translate_y_spin.value()
        self.path.rotate_deg = self.rotate_spin.value()
        super().accept()


class TrackTestDialog(QDialog):
    """Fenêtre plein écran pour tester un tracé sur piste modulaire."""

    def __init__(
        self,
        layer: LayerConfig,
        path: PathConfig,
        *,
        lang: str = "fr",
        points_per_path: int = 6000,
        parent=None,
    ):
        super().__init__(parent)
        self.lang = lang
        self.setWindowTitle(tr(self.lang, "track_test_title"))

        self.demo_widget = modular_track_demo.ModularTrackDemo(auto_start=False)
        self.points_per_path = max(2, int(points_per_path))

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        layout.addWidget(self.demo_widget, stretch=1)

        controls = QHBoxLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setSpacing(4)

        self.btn_start = QPushButton(tr(self.lang, "anim_start"))
        self.btn_reset = QPushButton(tr(self.lang, "anim_reset"))
        self.lbl_speed = QLabel(tr(self.lang, "anim_speed_label"))
        self.speed_spin = QDoubleSpinBox()
        self.speed_spin.setRange(0.0, 1_000_000.0)
        self.speed_spin.setDecimals(2)
        self.speed_spin.setSingleStep(0.25)
        self.speed_spin.setValue(1.0)
        self.speed_spin.setSpecialValueText(tr(self.lang, "anim_speed_infinite"))
        self.speed_spin.setSuffix(tr(self.lang, "anim_speed_suffix"))
        self.btn_half = QPushButton("/2")
        self.btn_double = QPushButton("x2")
        self.btn_close = QPushButton(tr(self.lang, "dlg_close"))

        controls.addWidget(self.btn_start)
        controls.addWidget(self.btn_reset)
        controls.addWidget(self.lbl_speed)
        controls.addWidget(self.speed_spin)
        controls.addWidget(self.btn_half)
        controls.addWidget(self.btn_double)
        controls.addStretch(1)
        controls.addWidget(self.btn_close)
        layout.addLayout(controls)

        self.btn_start.clicked.connect(self._toggle_animation)
        self.btn_reset.clicked.connect(self._reset_animation)
        self.speed_spin.valueChanged.connect(self._on_speed_changed)
        self.btn_half.clicked.connect(lambda: self._apply_speed_factor(0.5))
        self.btn_double.clicked.connect(lambda: self._apply_speed_factor(2.0))
        self.btn_close.clicked.connect(self.accept)

        self._apply_configuration(
            layer,
            path,
        )

    def _apply_configuration(
        self,
        layer: LayerConfig,
        path: PathConfig,
    ):
        if len(layer.gears) < 2:
            QMessageBox.information(
                self,
                tr(self.lang, "track_test_title"),
                tr(self.lang, "track_test_unavailable"),
            )
            return

        g0 = layer.gears[0]
        g1 = layer.gears[1]
        if g0.gear_type != "modulaire" or not getattr(g0, "modular_notation", ""):
            QMessageBox.information(
                self,
                tr(self.lang, "track_test_title"),
                tr(self.lang, "track_test_unavailable"),
            )
            return

        relation = g1.relation if g1.relation in ("dedans", "dehors") else "dedans"
        wheel_size = max(1, contact_size_for_relation(g1, relation))
        inner_size = g0.size if g0.size > 0 else 1
        outer_size = g0.outer_size if g0.outer_size > 0 else inner_size
        scale = getattr(layer, "zoom", 1.0) * getattr(path, "zoom", 1.0)

        self.demo_widget.set_configuration(
            notation=g0.modular_notation,
            wheel_size=wheel_size,
            hole_offset=path.hole_offset,
            relation=relation,
            phase_offset=getattr(path, "phase_offset", 0.0),
            inner_size=inner_size,
            outer_size=outer_size,
            steps=self.points_per_path,
            scale=scale,
        )
        if not self.demo_widget.stylo_points:
            QMessageBox.information(
                self,
                tr(self.lang, "track_test_title"),
                tr(self.lang, "track_test_unavailable"),
            )
            self.btn_start.setEnabled(False)
            self.btn_reset.setEnabled(False)
            return
        self._on_speed_changed(self.speed_spin.value())
        self.demo_widget.start_animation()
        self._update_start_button(self.demo_widget.timer.isActive())

    def _update_start_button(self, running: bool):
        self.btn_start.setText(
            tr(self.lang, "anim_pause") if running else tr(self.lang, "anim_start")
        )

    def _toggle_animation(self):
        if self.demo_widget.timer.isActive():
            self.demo_widget.stop_animation()
        else:
            self.demo_widget.start_animation()
        self._update_start_button(self.demo_widget.timer.isActive())

    def _reset_animation(self):
        self.demo_widget.reset_animation()
        self.demo_widget.stop_animation()
        self._update_start_button(False)

    def _apply_speed_factor(self, factor: float):
        val = self.speed_spin.value() * factor
        self.speed_spin.setValue(val)

    def _on_speed_changed(self, value: float):
        self.demo_widget.set_speed(value)
        is_running = value > 0.0 and self.demo_widget.timer.isActive()
        self._update_start_button(is_running)

# ---------- 6) Fenêtre superposée : gestion layers & paths ----------

class LayerManagerDialog(QDialog):
    """
    Fenêtre superposée pour gérer layers & paths.
    Affichage :
      - 3 colonnes : Nom | Type | Détails
    """

    def __init__(
        self,
        layers: List[LayerConfig],
        lang: str = "fr",
        parent=None,
        points_per_path: int = 6000,
    ):
        super().__init__(parent)
        self.lang = lang
        self.setWindowTitle(tr(self.lang, "dlg_layers_title"))
        self.resize(550, 500)

        self.layers: List[LayerConfig] = copy.deepcopy(layers)
        self.points_per_path: int = max(2, int(points_per_path))

        self.selected_layer_idx: int = 0
        self.selected_path_idx: Optional[int] = 0  # None = layer seul

        main_layout = QVBoxLayout(self)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels([
            tr(self.lang, "dlg_layers_col_name"),
            tr(self.lang, "dlg_layers_col_type"),
            tr(self.lang, "dlg_layers_col_details"),
        ])
        main_layout.addWidget(self.tree)

        btn_layout = QHBoxLayout()
        self.btn_add_layer = QPushButton(tr(self.lang, "dlg_layers_add_layer"))
        self.btn_add_path = QPushButton(tr(self.lang, "dlg_layers_add_path"))
        self.btn_edit = QPushButton(tr(self.lang, "dlg_layers_edit"))
        self.btn_toggle_enable = QPushButton()
        self.btn_toggle_paths = QPushButton()
        self.btn_move_up = QPushButton(tr(self.lang, "dlg_layers_move_up"))
        self.btn_move_down = QPushButton(tr(self.lang, "dlg_layers_move_down"))
        self.btn_test_track = QPushButton(tr(self.lang, "dlg_layers_test_track"))
        self.btn_remove = QPushButton(tr(self.lang, "dlg_layers_remove"))
        btn_layout.addWidget(self.btn_add_layer)
        btn_layout.addWidget(self.btn_add_path)
        btn_layout.addWidget(self.btn_edit)
        btn_layout.addWidget(self.btn_toggle_enable)
        btn_layout.addWidget(self.btn_toggle_paths)
        btn_layout.addWidget(self.btn_move_up)
        btn_layout.addWidget(self.btn_move_down)
        btn_layout.addWidget(self.btn_test_track)
        btn_layout.addWidget(self.btn_remove)
        main_layout.addLayout(btn_layout)

        bottom_layout = QHBoxLayout()
        self.btn_ok = QPushButton(tr(self.lang, "dlg_layers_ok"))
        self.btn_cancel = QPushButton(tr(self.lang, "dlg_layers_cancel"))
        bottom_layout.addWidget(self.btn_ok)
        bottom_layout.addWidget(self.btn_cancel)
        main_layout.addLayout(bottom_layout)

        self.btn_add_layer.clicked.connect(self.on_add_layer)
        self.btn_add_path.clicked.connect(self.on_add_path)
        self.btn_edit.clicked.connect(self.on_edit)
        self.btn_move_up.clicked.connect(self.on_move_up)
        self.btn_move_down.clicked.connect(self.on_move_down)
        self.btn_test_track.clicked.connect(self.on_test_track)
        self.btn_remove.clicked.connect(self.on_remove)
        self.btn_toggle_enable.clicked.connect(self.on_toggle_enable)
        self.btn_toggle_paths.clicked.connect(self.on_toggle_paths)
        self.btn_ok.clicked.connect(self.accept)
        self.btn_cancel.clicked.connect(self.reject)

        self.tree.currentItemChanged.connect(self.on_selection_changed)
        self.tree.itemDoubleClicked.connect(self.on_item_double_clicked)

        self.refresh_tree()

    # --- utilitaires ---

    def _layer_summary(self, layer: LayerConfig) -> str:
        parts = []
        gears_label = tr(self.lang, "layer_summary_gears_short")
        parts.append(f"{len(layer.gears)} {gears_label}, {tr(self.lang, 'layer_summary_zoom')} {layer.zoom:g}")
        gear_descs = []
        for i, g in enumerate(layer.gears):
            type_name = gear_type_label(g.gear_type, self.lang)
            if i == 0:
                type_name = type_name.capitalize()
            else:
                type_name = type_name.lower()

            if g.gear_type == "dsl" and getattr(g, "dsl_expression", None):
                size_str = g.dsl_expression
            elif g.gear_type in ("anneau", "modulaire") and g.outer_size > 0:
                size_str = f"{g.outer_size}/{g.size}"
            else:
                size_str = f"{g.size}"

            rel = relation_label(g.relation, self.lang)
            gear_descs.append(f"{type_name} {size_str} {rel}")
        if gear_descs:
            parts.append(", ".join(gear_descs))

        # Si le premier engrenage est modulaire, ajouter la notation
        if layer.gears and layer.gears[0].gear_type == "modulaire":
            notation = getattr(layer.gears[0], "modular_notation", None)
            if notation:
                parts.append(f"mod: {notation}")

        return ", ".join(parts)

    def _path_summary(self, path: PathConfig) -> str:
        return f"{path.hole_offset:g}, {path.phase_offset:g}, {path.color}, {path.stroke_width:g}, zoom {path.zoom:g}"

    def _layer_allows_test(self, layer: Optional[LayerConfig]) -> bool:
        if not layer or len(layer.gears) < 2:
            return False
        g0 = layer.gears[0]
        return g0.gear_type == "modulaire" and bool(
            getattr(g0, "modular_notation", "")
        )

    def _update_test_button_state(self):
        obj, kind = self.get_selected_object()
        enabled = False
        if kind == "path":
            layer = self.find_parent_layer(obj)
            enabled = (
                self._layer_allows_test(layer)
                and layer is not None
                and layer.enable
                and obj.enable
            )
        self.btn_test_track.setEnabled(enabled)

    def _update_move_buttons_state(self):
        obj, kind = self.get_selected_object()
        can_move_up = False
        can_move_down = False

        if kind == "layer":
            li = self.layers.index(obj)
            can_move_up = li > 0
            can_move_down = li < len(self.layers) - 1
        elif kind == "path":
            layer = self.find_parent_layer(obj)
            if layer:
                pi = layer.paths.index(obj)
                can_move_up = pi > 0
                can_move_down = pi < len(layer.paths) - 1

        self.btn_move_up.setEnabled(can_move_up)
        self.btn_move_down.setEnabled(can_move_down)

    def _visibility_icon(self, enabled: bool) -> QIcon:
        icon_name = "visibility" if enabled else "visibility-off"
        icon = QIcon.fromTheme(icon_name)
        if icon.isNull():
            fallback = QStyle.SP_DialogYesButton if enabled else QStyle.SP_DialogNoButton
            icon = self.style().standardIcon(fallback)
        return icon

    def _enabled_layers_count(self) -> int:
        return sum(1 for layer in self.layers if layer.enable)

    def _enabled_paths_count(self, layer: LayerConfig) -> int:
        return sum(1 for path in layer.paths if path.enable)

    def _is_last_enabled_layer(self, layer: LayerConfig) -> bool:
        return layer.enable and self._enabled_layers_count() == 1

    def _is_last_enabled_path_in_last_layer(
        self, layer: LayerConfig, path: PathConfig
    ) -> bool:
        return (
            path.enable
            and layer.enable
            and self._enabled_layers_count() == 1
            and self._enabled_paths_count(layer) == 1
        )

    def _update_enable_button_state(self):
        obj, kind = self.get_selected_object()
        if not obj:
            self.btn_toggle_enable.setEnabled(False)
            self.btn_toggle_enable.setText("")
            return
        if kind == "layer":
            if obj.enable:
                self.btn_toggle_enable.setText(tr(self.lang, "dlg_layers_disable_layer"))
                self.btn_toggle_enable.setEnabled(not self._is_last_enabled_layer(obj))
            else:
                self.btn_toggle_enable.setText(tr(self.lang, "dlg_layers_enable_layer"))
                self.btn_toggle_enable.setEnabled(True)
        elif kind == "path":
            layer = self.find_parent_layer(obj)
            if not layer:
                self.btn_toggle_enable.setEnabled(False)
                return
            if obj.enable:
                self.btn_toggle_enable.setText(tr(self.lang, "dlg_layers_disable_path"))
                self.btn_toggle_enable.setEnabled(
                    not self._is_last_enabled_path_in_last_layer(layer, obj)
                )
            else:
                self.btn_toggle_enable.setText(tr(self.lang, "dlg_layers_enable_path"))
                self.btn_toggle_enable.setEnabled(True)

    def _update_paths_toggle_button_state(self):
        obj, kind = self.get_selected_object()
        if not obj:
            self.btn_toggle_paths.setEnabled(False)
            self.btn_toggle_paths.setText("")
            return
        if kind == "layer":
            layer = obj
        elif kind == "path":
            layer = self.find_parent_layer(obj)
            if not layer:
                self.btn_toggle_paths.setEnabled(False)
                return
        else:
            self.btn_toggle_paths.setEnabled(False)
            return
        if not layer.paths:
            self.btn_toggle_paths.setEnabled(False)
            self.btn_toggle_paths.setText("")
            return

        any_disabled = any(not path.enable for path in layer.paths)
        if any_disabled:
            self.btn_toggle_paths.setText(tr(self.lang, "dlg_layers_enable_all_paths"))
            self.btn_toggle_paths.setEnabled(True)
            return

        self.btn_toggle_paths.setText(tr(self.lang, "dlg_layers_disable_all_paths"))
        if self._is_last_enabled_layer(layer):
            self.btn_toggle_paths.setEnabled(False)
            return
        self.btn_toggle_paths.setEnabled(True)

    def refresh_tree(self):
        self.tree.clear()
        current_item_to_select = None

        for li, layer in enumerate(self.layers):
            layer_item = QTreeWidgetItem(
                [layer.name, tr(self.lang, "tree_type_layer"), self._layer_summary(layer)]
            )
            layer_item.setData(0, Qt.UserRole, layer)
            layer_item.setIcon(0, self._visibility_icon(layer.enable))
            self.tree.addTopLevelItem(layer_item)

            if li == self.selected_layer_idx and self.selected_path_idx is None:
                current_item_to_select = layer_item

            for pi, path in enumerate(layer.paths):
                path_item = QTreeWidgetItem(
                    [path.name, tr(self.lang, "tree_type_path"), self._path_summary(path)]
                )
                path_item.setData(0, Qt.UserRole, path)
                path_item.setIcon(0, self._visibility_icon(path.enable))
                layer_item.addChild(path_item)

                if (
                    li == self.selected_layer_idx
                    and self.selected_path_idx is not None
                    and pi == self.selected_path_idx
                ):
                    current_item_to_select = path_item

            layer_item.setExpanded(True)

        if self.selected_layer_idx >= len(self.layers):
            self.selected_layer_idx = max(0, len(self.layers) - 1)
            self.selected_path_idx = None

        if not current_item_to_select and self.tree.topLevelItemCount() > 0:
            current_item_to_select = self.tree.topLevelItem(self.selected_layer_idx)

        if current_item_to_select:
            self.tree.setCurrentItem(current_item_to_select)
        self._update_test_button_state()
        self._update_move_buttons_state()
        self._update_enable_button_state()
        self._update_paths_toggle_button_state()

    def get_selected_object(self):
        item = self.tree.currentItem()
        if not item:
            return None, None
        obj = item.data(0, Qt.UserRole)
        if isinstance(obj, LayerConfig):
            return obj, "layer"
        if isinstance(obj, PathConfig):
            return obj, "path"
        return None, None

    def find_parent_layer(self, path_obj: PathConfig) -> Optional[LayerConfig]:
        for layer in self.layers:
            if path_obj in layer.paths:
                return layer
        return None

    # --- gestion sélection & double-clic ---

    def on_selection_changed(
        self,
        current: Optional[QTreeWidgetItem],
        previous: Optional[QTreeWidgetItem],
    ):
        if current is None:
            self.btn_test_track.setEnabled(False)
            self.btn_move_up.setEnabled(False)
            self.btn_move_down.setEnabled(False)
            self._update_enable_button_state()
            self._update_paths_toggle_button_state()
            return

        obj = current.data(0, Qt.UserRole)
        if isinstance(obj, LayerConfig):
            for li, layer in enumerate(self.layers):
                if layer is obj:
                    self.selected_layer_idx = li
                    self.selected_path_idx = None
                    break

        elif isinstance(obj, PathConfig):
            layer = self.find_parent_layer(obj)
            if not layer:
                return
            li = self.layers.index(layer)
            pi = layer.paths.index(obj)
            self.selected_layer_idx = li
            self.selected_path_idx = pi
        self._update_test_button_state()
        self._update_move_buttons_state()
        self._update_enable_button_state()
        self._update_paths_toggle_button_state()

    def on_item_double_clicked(self, item: QTreeWidgetItem, column: int):
        self.on_edit()

    # --- callbacks boutons ---

    def on_add_layer(self):
        g0 = GearConfig(
            name=tr(self.lang, "default_ring_name"),
            gear_type="anneau",
            size=105,
            outer_size=150,
            relation="stationnaire",
        )
        g1 = GearConfig(
            name=tr(self.lang, "default_wheel_name"),
            gear_type="roue",
            size=30,
            relation="dedans",
        )
        new_layer = LayerConfig(
            name=f"{tr(self.lang, 'default_layer_name')} {len(self.layers) + 1}",
            enable=True,
            zoom=1.0,
            gears=[g0, g1],
            paths=[
                PathConfig(
                    name=f"{tr(self.lang, 'default_path_name')} 1",
                    hole_offset=1.0,
                    zoom=1.0,
                    enable=True,
                )
            ],
        )
        self.layers.append(new_layer)

        self.selected_layer_idx = len(self.layers) - 1
        self.selected_path_idx = None
        self.refresh_tree()

    def on_add_path(self):
        obj, kind = self.get_selected_object()
        if kind == "layer":
            layer = obj
        elif kind == "path":
            layer = self.find_parent_layer(obj)
        else:
            QMessageBox.warning(
                self,
                tr(self.lang, "dlg_layers_need_layer_title"),
                tr(self.lang, "dlg_layers_need_layer_text"),
            )
            return
        new_path = PathConfig(
            name=f"{tr(self.lang, 'default_path_name')} {len(layer.paths) + 1}",
            hole_offset=1.0,
            zoom=1.0,
            enable=layer.enable,
        )
        layer.paths.append(new_path)

        li = self.layers.index(layer)
        pi = len(layer.paths) - 1
        self.selected_layer_idx = li
        self.selected_path_idx = pi

        self.refresh_tree()

    def on_edit(self):
        obj, kind = self.get_selected_object()
        if not obj:
            return
        if kind == "layer":
            dlg = LayerEditDialog(
                obj,
                lang=self.lang,
                parent=self,
            )
        else:
            dlg = PathEditDialog(obj, lang=self.lang, parent=self)
        if dlg.exec() == QDialog.Accepted:
            self.refresh_tree()

    def on_toggle_enable(self):
        obj, kind = self.get_selected_object()
        if not obj:
            return
        if kind == "layer":
            if obj.enable:
                if self._is_last_enabled_layer(obj):
                    return
                obj.enable = False
            else:
                obj.enable = True
        elif kind == "path":
            layer = self.find_parent_layer(obj)
            if not layer:
                return
            if obj.enable:
                if self._is_last_enabled_path_in_last_layer(layer, obj):
                    return
                obj.enable = False
                if not any(path.enable for path in layer.paths):
                    layer.enable = False
            else:
                obj.enable = True
        self.refresh_tree()

    def on_toggle_paths(self):
        obj, kind = self.get_selected_object()
        if not obj:
            return
        if kind == "layer":
            layer = obj
        elif kind == "path":
            layer = self.find_parent_layer(obj)
        else:
            return
        if not layer:
            return
        if not layer.paths:
            return

        any_disabled = any(not path.enable for path in layer.paths)
        if any_disabled:
            for path in layer.paths:
                path.enable = True
        else:
            if self._is_last_enabled_layer(layer):
                return
            for path in layer.paths:
                path.enable = False
            if not any(path.enable for path in layer.paths):
                layer.enable = False
        self.refresh_tree()

    def on_move_up(self):
        obj, kind = self.get_selected_object()
        if kind == "layer":
            li = self.layers.index(obj)
            if li > 0:
                self.layers[li - 1], self.layers[li] = self.layers[li], self.layers[li - 1]
                self.selected_layer_idx = li - 1
                self.selected_path_idx = None
        elif kind == "path":
            layer = self.find_parent_layer(obj)
            if layer:
                pi = layer.paths.index(obj)
                if pi > 0:
                    layer.paths[pi - 1], layer.paths[pi] = layer.paths[pi], layer.paths[pi - 1]
                    self.selected_layer_idx = self.layers.index(layer)
                    self.selected_path_idx = pi - 1

        self.refresh_tree()

    def on_move_down(self):
        obj, kind = self.get_selected_object()
        if kind == "layer":
            li = self.layers.index(obj)
            if li < len(self.layers) - 1:
                self.layers[li], self.layers[li + 1] = self.layers[li + 1], self.layers[li]
                self.selected_layer_idx = li + 1
                self.selected_path_idx = None
        elif kind == "path":
            layer = self.find_parent_layer(obj)
            if layer:
                pi = layer.paths.index(obj)
                if pi < len(layer.paths) - 1:
                    layer.paths[pi], layer.paths[pi + 1] = layer.paths[pi + 1], layer.paths[pi]
                    self.selected_layer_idx = self.layers.index(layer)
                    self.selected_path_idx = pi + 1

        self.refresh_tree()

    def on_test_track(self):
        obj, kind = self.get_selected_object()
        if kind != "path":
            QMessageBox.information(
                self,
                tr(self.lang, "track_test_title"),
                tr(self.lang, "track_test_unavailable"),
            )
            return

        layer = self.find_parent_layer(obj)
        if not self._layer_allows_test(layer):
            QMessageBox.information(
                self,
                tr(self.lang, "track_test_title"),
                tr(self.lang, "track_test_unavailable"),
            )
            return

        dlg = TrackTestDialog(
            layer,
            obj,
            lang=self.lang,
            parent=self,
            points_per_path=self.points_per_path,
        )
        dlg.setWindowState(dlg.windowState() | Qt.WindowFullScreen)
        dlg.exec()

    def on_remove(self):
        obj, kind = self.get_selected_object()
        if not obj:
            return

        if kind == "layer":
            if len(self.layers) == 1:
                QMessageBox.warning(
                    self,
                    tr(self.lang, "dlg_layers_must_keep_layer_title"),
                    tr(self.lang, "dlg_layers_must_keep_layer_text"),
                )
                return
            li = self.layers.index(obj)
            del self.layers[li]
            if li >= len(self.layers):
                li = len(self.layers) - 1
            self.selected_layer_idx = max(li, 0)
            self.selected_path_idx = None

        else:  # path
            layer = self.find_parent_layer(obj)
            if layer is not None:
                li = self.layers.index(layer)
                pi = layer.paths.index(obj)
                if len(layer.paths) == 1:
                    ret = QMessageBox.question(
                        self,
                        tr(self.lang, "dlg_layers_remove_last_path_title"),
                        tr(self.lang, "dlg_layers_remove_last_path_text"),
                        QMessageBox.Yes | QMessageBox.No,
                    )
                    if ret != QMessageBox.Yes:
                        return
                del layer.paths[pi]
                if layer.paths:
                    pi = min(pi, len(layer.paths) - 1)
                    self.selected_layer_idx = li
                    self.selected_path_idx = pi
                else:
                    self.selected_layer_idx = li
                    self.selected_path_idx = None

        if self.layers and not any(layer.enable for layer in self.layers):
            self.layers[0].enable = True
            for path in self.layers[0].paths:
                path.enable = True

        self.refresh_tree()

    def get_layers(self) -> List[LayerConfig]:
        return self.layers

class ModularTrackView(QWidget):
    """
    Widget de visualisation d'une piste modulaire :
      - polyline centrale (modular_tracks)
      - bande de largeur réelle (inner/outer)
      - graduations approximatives le long de la piste
      - ligne rouge perpendiculaire à la tangente de fin
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.points: List[Tuple[float, float]] = []
        self.have_track = False
        self.inner_size = 96
        self.outer_size = 144
        self.total_length = 0.0
        self.segments = []
        self.last_tangent = 0.0  # angle de la tangente au dernier point (rad)

    def sizeHint(self):
        return QSize(500, 500)

    def clear_track(self):
        self.points = []
        self.segments = []
        self.have_track = False
        self.total_length = 0.0
        self.last_tangent = 0.0
        self.update()

    def set_track(
        self,
        track: modular_tracks.TrackBuildResult,
        inner_size: int,
        outer_size: int,
    ):
        self.points = track.points or []
        self.segments = track.segments or []
        self.total_length = track.total_length
        self.inner_size = max(1, inner_size)
        self.outer_size = max(self.inner_size + 1, outer_size)
        self.have_track = len(self.points) > 1 and self.total_length > 0.0
        if self.have_track and self.segments:
            _, theta, _ = modular_tracks._interpolate_on_segments(
                self.total_length, self.segments
            )
            self.last_tangent = theta
        else:
            self.last_tangent = 0.0
        self.update()

    def _compute_scale(self, w: int, h: int) -> float:
        if not self.points:
            return 1.0
        max_x = max(abs(x) for x, _ in self.points) or 1.0
        max_y = max(abs(y) for _, y in self.points) or 1.0
        sx = (w * 0.45) / max_x
        sy = (h * 0.45) / max_y
        return min(sx, sy)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        w = self.width()
        h = self.height()

        painter.fillRect(self.rect(), self.palette().window())

        if not self.have_track or not self.points:
            painter.setPen(Qt.gray)
            txt = "Notation incomplète" if self.parent() is None else ""
            if txt:
                painter.drawText(self.rect(), Qt.AlignCenter, txt)
            return

        # Échelle et transform pour centrer autour de (0,0) et inverser Y
        scale = self._compute_scale(w, h)
        painter.translate(w / 2.0, h / 2.0)
        painter.scale(scale, -scale)

        # Largeur réelle de la piste (unités abstraites)
        inner_r = float(self.inner_size) / (2.0 * math.pi)
        outer_r = float(self.outer_size) / (2.0 * math.pi)
        width_mm = max(outer_r - inner_r, UNIT_LENGTH)
        half_w = width_mm / 2.0

        # 1) polyline centrale
        pen_center = QPen(QColor("#606060"))
        pen_center.setWidthF(0)  # ligne "cosmétique"
        painter.setPen(pen_center)

        for i in range(len(self.points) - 1):
            x0, y0 = self.points[i]
            x1, y1 = self.points[i + 1]
            painter.drawLine(QPointF(x0, y0), QPointF(x1, y1))

        # 2) bords de la piste (inner/outer approximatifs)
        pen_border = QPen(QColor("#808080"))
        pen_border.setWidthF(0)
        painter.setPen(pen_border)

        for i in range(len(self.points) - 1):
            x0, y0 = self.points[i]
            x1, y1 = self.points[i + 1]
            dx = x1 - x0
            dy = y1 - y0
            seg_len = math.hypot(dx, dy) or 1.0
            nx = -dy / seg_len
            ny = dx / seg_len

            ix0 = x0 - nx * half_w
            iy0 = y0 - ny * half_w
            ix1 = x1 - nx * half_w
            iy1 = y1 - ny * half_w

            ox0 = x0 + nx * half_w
            oy0 = y0 + ny * half_w
            ox1 = x1 + nx * half_w
            oy1 = y1 + ny * half_w

            painter.drawLine(QPointF(ix0, iy0), QPointF(ix1, iy1))
            painter.drawLine(QPointF(ox0, oy0), QPointF(ox1, oy1))

        # 3) graduations (petits ticks côté "outer")
        L = self.total_length if self.have_track else 0.0
        if L > 0.0 and self.segments:
            pen_ticks = QPen(QColor("#404040"))
            pen_ticks.setWidthF(0)
            painter.setPen(pen_ticks)
            tick_len = width_mm * 0.4
            num_ticks = max(1, int(L))
            for k in range(num_ticks):
                s = (k + 0.5)
                (x, y), theta, _ = modular_tracks._interpolate_on_segments(
                    s % L, self.segments
                )
                nx = -math.sin(theta)
                ny = math.cos(theta)
                bx = x + nx * half_w
                by = y + ny * half_w
                tx = bx + nx * tick_len
                ty = by + ny * tick_len
                painter.drawLine(QPointF(bx, by), QPointF(tx, ty))

        # 4) Ligne rouge perpendiculaire à la tangente de fin
        if len(self.points) >= 2:
            x_last, y_last = self.points[-1]
            theta = self.last_tangent
            nx = -math.sin(theta)
            ny = math.cos(theta)
            line_len = width_mm * 1.5  # 50 % de plus que la largeur
            hx = (line_len / 2.0) * nx
            hy = (line_len / 2.0) * ny

            pen_red = QPen(QColor("#ff0000"))
            pen_red.setWidthF(0)
            painter.setPen(pen_red)
            painter.drawLine(
                QPointF(x_last - hx, y_last - hy),
                QPointF(x_last + hx, y_last + hy),
            )

        painter.end()

class ModularTrackEditorDialog(QDialog):
    """
    Éditeur en temps réel pour les pistes modulaires :
      - notation (avec surlignage de la partie comprise)
      - paramètres inner/outer
      - vue centrée sur le barycentre (via modular_tracks)
    """

    def __init__(
        self,
        lang: str = "fr",
        parent=None,
        initial_notation: str = "",
        inner_size: int = 96,
        outer_size: int = 144,
    ):
        super().__init__(parent)
        self.lang = lang
        self.setWindowTitle(tr(self.lang, "mod_editor_title"))
        self.resize(900, 600)

        self._initial_notation = initial_notation or ""

        self.track_view = ModularTrackView(self)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(4)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(4)

        # --- Zone notation + paramètres ---
        top_layout = QHBoxLayout()
        notation_layout = QVBoxLayout()

        lbl_notation = QLabel(tr(self.lang, "mod_editor_notation_label"))
        self.notation_edit = QLineEdit()
        self.notation_display = QLabel()
        self.notation_display.setTextFormat(Qt.RichText)
        self.notation_display.setMinimumHeight(24)

        notation_layout.addWidget(lbl_notation)
        notation_layout.addWidget(self.notation_edit)
        notation_layout.addWidget(self.notation_display)

        params_layout = QHBoxLayout()
        self.inner_spin = QSpinBox()
        self.inner_spin.setRange(1, 2000)
        self.inner_spin.setValue(max(1, inner_size))

        self.outer_spin = QSpinBox()
        self.outer_spin.setRange(1, 4000)
        self.outer_spin.setValue(max(1, outer_size))

        params_layout.addWidget(QLabel(tr(self.lang, "mod_editor_inner_size")))
        params_layout.addWidget(self.inner_spin)
        params_layout.addWidget(QLabel(tr(self.lang, "mod_editor_outer_size")))
        params_layout.addWidget(self.outer_spin)
        params_layout.addStretch(1)

        notation_layout.addLayout(params_layout)

        top_layout.addLayout(notation_layout, 2)
        top_layout.addWidget(self.track_view, 3)

        main_layout.addLayout(top_layout)

        # Info sur la piste
        self.info_label = QLabel()
        main_layout.addWidget(self.info_label)

        # Boutons OK/Annuler
        btn_layout = QHBoxLayout()
        btn_ok = QPushButton(tr(self.lang, "dlg_ok"))
        btn_cancel = QPushButton(tr(self.lang, "dlg_cancel"))
        btn_ok.clicked.connect(self.accept)
        btn_cancel.clicked.connect(self.reject)
        btn_layout.addStretch(1)
        btn_layout.addWidget(btn_ok)
        btn_layout.addWidget(btn_cancel)
        main_layout.addLayout(btn_layout)

        # Connexions
        self.notation_edit.textChanged.connect(self.update_track)
        self.inner_spin.valueChanged.connect(self.update_track)
        self.outer_spin.valueChanged.connect(self.update_track)

        # Valeurs initiales
        self.notation_edit.setText(self._initial_notation)
        self.update_track()

    def result_notation(self) -> str:
        return self.notation_edit.text().strip()

    def update_track(self):
        text = self.notation_edit.text()
        valid, rest, has_piece = split_valid_modular_notation(text)

        # Construction du texte surligné (tout en MAJUSCULES, sans espaces)
        if valid or rest:
            html = ""
            if valid:
                html += (
                    "<span style='background-color:#404000;color:#ffffa0;'>"
                    + escape(valid)
                    + "</span>"
                )
            if rest:
                html += (
                    "<span style='color:#808080;'>"
                    + escape(rest)
                    + "</span>"
                )
            self.notation_display.setText(html)
        else:
            self.notation_display.setText("&nbsp;")

        inner_size = self.inner_spin.value()
        outer_size = self.outer_spin.value()

        # Rien à dessiner si aucune pièce
        if not has_piece or not valid:
            self.track_view.clear_track()
            self.info_label.setText(tr(self.lang, "mod_editor_info_no_piece"))
            return

        try:
            track = modular_tracks.build_track_from_notation(
                valid,
                inner_size=inner_size,
                outer_size=outer_size,
                steps_per_unit=3,
            )
        except NotImplementedError as e:
            # Cas des pièces non encore gérées (Y, etc.)
            self.track_view.clear_track()
            self.info_label.setText(
                tr(self.lang, "mod_editor_info_error").format(error=e)
            )
            return
        except Exception as e:
            self.track_view.clear_track()
            self.info_label.setText(
                tr(self.lang, "mod_editor_info_error").format(error=e)
            )
            return

        if len(track.points) < 2 or track.total_length <= 0.0:
            self.track_view.clear_track()
            self.info_label.setText(tr(self.lang, "mod_editor_info_empty"))
            return

        self.track_view.set_track(track, inner_size, outer_size)
        inner_len, mid_len, outer_len = modular_tracks.compute_track_lengths(
            track, inner_size, outer_size
        )
        summaries = [
            tr(self.lang, "mod_editor_summary_inner").format(length=inner_len),
            tr(self.lang, "mod_editor_summary_mid").format(length=mid_len),
            tr(self.lang, "mod_editor_summary_outer").format(length=outer_len),
        ]
        self.info_label.setText("\n".join(summaries))

# ---------- 7) Fenêtre principale ----------

class SpiroWindow(QWidget):
    def __init__(self):
        super().__init__()

        # Langue par défaut : français
        self.language = localisation.resolve_language("fr")

        # Indicateur de restauration de géométrie
        self._geometry_restored = False

        # Couleur de fond
        self.bg_color: str = "#ffffff"

        # Taille du canevas (en pixels) et nombre de points par tracé
        self.canvas_width: int = 1000
        self.canvas_height: int = 1000
        self.points_per_path: int = 6000

        # Affichage optionnel
        self.animation_enabled: bool = True
        self.show_track: bool = True

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # ----- Barre de menus -----
        menubar = QMenuBar()
        self.menu_bar = menubar

        # Menu Fichier
        self.menu_file = QMenu(menubar)
        menubar.addMenu(self.menu_file)

        self.act_load_json = QAction(menubar)
        self.act_save_json = QAction(menubar)
        self.act_export_svg = QAction(menubar)
        self.act_export_png = QAction(menubar)
        self.act_quit = QAction(menubar)

        self.menu_file.addAction(self.act_load_json)
        self.menu_file.addAction(self.act_save_json)
        self.menu_file.addSeparator()
        self.menu_file.addAction(self.act_export_svg)
        self.menu_file.addAction(self.act_export_png)
        self.menu_file.addSeparator()
        self.menu_file.addAction(self.act_quit)

        self.act_load_json.triggered.connect(self.load_from_json)
        self.act_save_json.triggered.connect(self.save_to_json)
        self.act_export_svg.triggered.connect(self.export_svg)
        self.act_export_png.triggered.connect(self.export_png)
        self.act_quit.triggered.connect(self.close)

        # Menu Couches
        self.menu_layers = QMenu(menubar)
        menubar.addMenu(self.menu_layers)
        self.act_manage_layers = QAction(menubar)
        self.menu_layers.addAction(self.act_manage_layers)
        self.act_manage_layers.triggered.connect(self.open_layer_manager)

        # Menu Options
        self.menu_options = QMenu(menubar)
        menubar.addMenu(self.menu_options)
        self.act_bg_color = QAction(menubar)
        self.act_bg_color.triggered.connect(self.edit_bg_color)
        self.menu_options.addAction(self.act_bg_color)

        self.act_shape_lab = QAction(menubar)
        self.act_shape_lab.triggered.connect(self.open_shape_lab)
        self.menu_options.addAction(self.act_shape_lab)

        # NOUVELLE OPTION : taille du canevas et précision
        self.act_canvas = QAction(menubar)
        self.act_canvas.triggered.connect(self.edit_canvas_settings)
        self.menu_options.addAction(self.act_canvas)

        # Sous-menu Langue
        self.menu_lang = QMenu(menubar)
        self.menu_options.addMenu(self.menu_lang)
        self.language_actions: Dict[str, QAction] = {}
        self._build_language_menu(menubar)


        # Menu Régénérer
        self.menu_regen = QMenu(menubar)
        menubar.addMenu(self.menu_regen)
        self.act_animation_enabled = QAction(menubar)
        self.act_animation_enabled.setCheckable(True)
        self.act_animation_enabled.setChecked(True)
        self.act_animation_enabled.triggered.connect(self._set_animation_enabled)
        self.menu_regen.addAction(self.act_animation_enabled)

        self.act_show_track = QAction(menubar)
        self.act_show_track.setCheckable(True)
        self.act_show_track.setChecked(True)
        self.act_show_track.triggered.connect(self._set_show_track)
        self.menu_regen.addAction(self.act_show_track)

        self.menu_regen.addSeparator()
        self.act_regen = QAction(menubar)
        self.act_regen.triggered.connect(self.update_svg)
        self.menu_regen.addAction(self.act_regen)

        # Menu Aide
        self.menu_help = QMenu(menubar)
        menubar.addMenu(self.menu_help)
        self.act_help_manual = QAction(menubar)
        self.act_help_about = QAction(menubar)
        self.menu_help.addAction(self.act_help_manual)
        self.menu_help.addAction(self.act_help_about)
        self.act_help_manual.triggered.connect(self.open_manual)
        self.act_help_about.triggered.connect(self.show_about)

        main_layout.addWidget(menubar)

        self.svg_widget = QSvgWidget()
        self.svg_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        svg_container = QWidget()
        svg_container_layout = QHBoxLayout(svg_container)
        svg_container_layout.setContentsMargins(0, 0, 0, 0)
        svg_container_layout.setSpacing(0)
        svg_container_layout.addWidget(
            self.svg_widget, alignment=Qt.AlignmentFlag.AlignCenter
        )
        self.svg_container = svg_container
        main_layout.addWidget(svg_container, stretch=1)

        # ----- Animation du tracé -----
        self._last_svg_data: Optional[str] = None
        self._animation_render_data = None
        self._animation_timer = QTimer(self)
        self._animation_timer.timeout.connect(self._on_animation_tick)
        self._animation_running = False
        self._animation_progress = 0.0
        self._animation_last_time: Optional[float] = None
        self._animation_speed = 1.0

        anim_layout = QHBoxLayout()
        anim_layout.setContentsMargins(0, 0, 0, 0)
        anim_layout.setSpacing(4)
        self.anim_start_btn = QPushButton(tr(self.language, "anim_start"))
        self.anim_reset_btn = QPushButton(tr(self.language, "anim_reset"))
        self.anim_speed_label = QLabel(tr(self.language, "anim_speed_label"))
        self.anim_speed_spin = QDoubleSpinBox()
        self.anim_speed_spin.setRange(0.0, 1_000_000.0)
        self.anim_speed_spin.setDecimals(2)
        self.anim_speed_spin.setSingleStep(0.25)
        self.anim_speed_spin.setValue(1.0)
        self.anim_speed_spin.setSpecialValueText(tr(self.language, "anim_speed_infinite"))
        self.anim_speed_spin.setSuffix(tr(self.language, "anim_speed_suffix"))
        self.anim_btn_half = QPushButton("/2")
        self.anim_btn_double = QPushButton("x2")

        self.anim_start_btn.clicked.connect(self._toggle_animation)
        self.anim_reset_btn.clicked.connect(self._reset_animation)
        self.anim_speed_spin.valueChanged.connect(self._on_anim_speed_changed)
        self.anim_btn_half.clicked.connect(lambda: self._apply_anim_speed_factor(0.5))
        self.anim_btn_double.clicked.connect(lambda: self._apply_anim_speed_factor(2.0))

        anim_layout.addWidget(self.anim_start_btn)
        anim_layout.addWidget(self.anim_reset_btn)
        anim_layout.addWidget(self.anim_speed_label)
        anim_layout.addWidget(self.anim_speed_spin)
        anim_layout.addWidget(self.anim_btn_half)
        anim_layout.addWidget(self.anim_btn_double)
        anim_layout.addStretch(1)
        anim_container = QWidget()
        anim_container.setLayout(anim_layout)
        self.anim_container = anim_container
        self.anim_container.setVisible(self.animation_enabled)
        main_layout.addWidget(anim_container)

        # Layer par défaut : anneau 150/105 + roue 30 dedans
        g0 = GearConfig(
            name=tr(self.language, "default_ring_name"),
            gear_type="anneau",
            size=105,        # intérieur
            outer_size=150,  # extérieur
            relation="stationnaire",
        )
        g1 = GearConfig(
            name=tr(self.language, "default_wheel_name"),
            gear_type="roue",
            size=30,
            relation="dedans",
        )
        base_layer = LayerConfig(
            name=f"{tr(self.language, 'default_layer_name')} 1",
            enable=True,
            zoom=1.0,
            gears=[g0, g1],
            paths=[
                PathConfig(
                    name=f"{tr(self.language, 'default_path_name')} 1",
                    enable=True,
                    hole_offset=1.0,
                    phase_offset=0.0,
                    color="red",
                    stroke_width=1.2,
                    zoom=1.0,
                ),
                PathConfig(
                    name=f"{tr(self.language, 'default_path_name')} 2",
                    enable=True,
                    hole_offset=2.0,
                    phase_offset=0.25,
                    color="#0000aa",
                    stroke_width=1.0,
                    zoom=1.0,
                ),
            ],
        )
        self.layers: List[LayerConfig] = [base_layer]

        self.setLayout(main_layout)

        loaded_from_disk = self._load_persisted_state()

        self._update_animation_controls()
        self._update_svg_size()

        # Appliquer la langue et générer le premier SVG si rien n'a été chargé
        if not loaded_from_disk:
            self.apply_language()
            self.update_svg()

    # ----- Langue -----

    def set_language(self, lang: str):
        self.language = localisation.resolve_language(lang or "fr")
        self.apply_language()

    def apply_language(self):
        self.setWindowTitle(tr(self.language, "app_title"))

        # Menus
        self.menu_file.setTitle(tr(self.language, "menu_file"))
        self.menu_layers.setTitle(tr(self.language, "menu_layers"))
        self.menu_options.setTitle(tr(self.language, "menu_options"))
        self.menu_regen.setTitle(tr(self.language, "menu_regen"))
        self.menu_help.setTitle(tr(self.language, "menu_help"))

        # Actions Fichier
        self.act_load_json.setText(tr(self.language, "menu_file_load_json"))
        self.act_save_json.setText(tr(self.language, "menu_file_save_json"))
        self.act_export_svg.setText(tr(self.language, "menu_file_export_svg"))
        self.act_export_png.setText(tr(self.language, "menu_file_export_png"))
        self.act_quit.setText(tr(self.language, "menu_file_quit"))

        # Actions Options
        self.act_manage_layers.setText(tr(self.language, "menu_layers_manage"))
        self.act_bg_color.setText(tr(self.language, "menu_options_bgcolor"))
        self.act_shape_lab.setText(tr(self.language, "menu_options_shape_lab"))
        self.act_canvas.setText(tr(self.language, "menu_options_canvas"))
        self.menu_lang.setTitle(tr(self.language, "menu_options_language"))
        self.act_animation_enabled.setText(tr(self.language, "menu_regen_animation"))
        self.act_show_track.setText(tr(self.language, "menu_regen_show_track"))
        self.act_regen.setText(tr(self.language, "menu_regen_draw"))
        self.act_help_manual.setText(tr(self.language, "menu_help_manual"))
        self.act_help_about.setText(tr(self.language, "menu_help_about"))

        self._refresh_animation_texts()

        # Checkmarks langue
        for code, action in self.language_actions.items():
            action.setText(localisation.language_display_name(code))
            action.setChecked(code == self.language)

    def _available_svg_space(self) -> Tuple[int, int]:
        layout = self.layout()
        if not layout:
            return 0, 0
        margins = layout.contentsMargins()
        spacing = layout.spacing()
        available_width = max(0, self.width() - margins.left() - margins.right())
        available_height = max(0, self.height() - margins.top() - margins.bottom())

        menu_height = max(self.menu_bar.height(), self.menu_bar.sizeHint().height())
        available_height = max(0, available_height - menu_height)

        anim_height = 0
        if getattr(self, "anim_container", None) is not None:
            anim_height = max(
                self.anim_container.height(), self.anim_container.sizeHint().height()
            )
        available_height = max(0, available_height - anim_height)

        # Two spacings separate menubar/SVG and SVG/controls
        available_height = max(0, available_height - (spacing * 2))

        # Account for the SVG container margins, which stay at 0 but keep logic consistent
        if getattr(self, "svg_container", None) is not None:
            svg_margins = self.svg_container.layout().contentsMargins()
            available_width = max(
                0, available_width - svg_margins.left() - svg_margins.right()
            )
            available_height = max(
                0, available_height - svg_margins.top() - svg_margins.bottom()
            )

        return available_width, available_height

    def _build_language_menu(self, menubar: QMenuBar):
        self.menu_lang.clear()
        self.language_actions.clear()
        for code in localisation.available_languages():
            action = QAction(menubar)
            action.setCheckable(True)
            action.setData(code)
            action.triggered.connect(lambda _checked=False, lang=code: self.set_language(lang))
            self.menu_lang.addAction(action)
            self.language_actions[code] = action

    def _resolve_repo_info(self) -> Tuple[str, Optional[str]]:
        repo_root = os.path.dirname(os.path.abspath(__file__))
        try:
            result = subprocess.run(
                ["git", "-C", repo_root, "rev-parse", "--abbrev-ref", "HEAD"],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except Exception:
            return GITHUB_REPO_URL, None
        branch = result.stdout.strip()
        if not branch or branch == "HEAD":
            return GITHUB_REPO_URL, None
        return f"{GITHUB_REPO_URL}/tree/{branch}", branch

    def open_manual(self):
        readme_path = localisation.resolve_readme_path(self.language)
        repo_root = Path(os.path.abspath(__file__)).parent
        readme = readme_path.relative_to(repo_root).as_posix()
        _repo_url, branch = self._resolve_repo_info()
        ref = branch or "main"
        url = f"{GITHUB_REPO_URL}/blob/{ref}/{readme}"
        QDesktopServices.openUrl(QUrl(url))

    def show_about(self):
        url, _branch = self._resolve_repo_info()
        text = (
            "<p><b>Spiro Sim</b></p>"
            "<p>Créé par Alyndiar</p>"
            f"<p>Version {APP_VERSION}</p>"
            "<p>CC-BY-SA 4.0</p>"
            f'<p><a href="{url}">{url}</a></p>'
        )
        dlg = QMessageBox(self)
        dlg.setWindowTitle(tr(self.language, "dlg_about_title"))
        dlg.setTextFormat(Qt.RichText)
        dlg.setTextInteractionFlags(Qt.TextBrowserInteraction)
        dlg.setText(text)
        dlg.setStandardButtons(QMessageBox.Ok)
        dlg.exec()

    def _update_svg_size(self):
        available_width, available_height = self._available_svg_space()
        square_size = max(50, min(available_width, available_height))
        self.svg_widget.setFixedSize(QSize(square_size, square_size))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_svg_size()

    # ----- SVG -----

    def update_svg(self):
        bg_norm = normalize_color_string(self.bg_color) or "#ffffff"
        result = layers_to_svg(
            self.layers,
            width=self.canvas_width,
            height=self.canvas_height,
            bg_color=bg_norm,
            points_per_path=self.points_per_path,
            show_tracks=self.show_track,
            return_render_data=True,
        )
        if isinstance(result, tuple):
            svg_data, render_data = result
        else:
            svg_data, render_data = result, None
        self._last_svg_data = svg_data
        self._animation_render_data = render_data
        self._reset_animation_state()
        self.load_svg(svg_data)

    def load_svg(self, svg_string: str):
        data = QByteArray(svg_string.encode("utf-8"))
        self.svg_widget.load(data)

    # ----- Animation -----

    def _set_animation_enabled(self, enabled: bool):
        self.animation_enabled = bool(enabled)
        if self.act_animation_enabled.isChecked() != self.animation_enabled:
            self.act_animation_enabled.setChecked(self.animation_enabled)
        if not self.animation_enabled:
            self._stop_animation()
        self.anim_container.setVisible(self.animation_enabled)
        self._update_animation_controls()

    def _set_show_track(self, checked: bool, trigger_update: bool = True):
        self.show_track = bool(checked)
        if self.act_show_track.isChecked() != self.show_track:
            self.act_show_track.setChecked(self.show_track)
        if trigger_update:
            self.update_svg()

    def _max_animation_points(self) -> int:
        if not self._animation_render_data:
            return 0
        paths = self._animation_render_data.get("paths") or []
        if not paths:
            return 0
        return max(len(p.get("points", [])) for p in paths)

    def _refresh_animation_texts(self):
        start_key = "anim_pause" if self._animation_running else "anim_start"
        self.anim_start_btn.setText(tr(self.language, start_key))
        self.anim_reset_btn.setText(tr(self.language, "anim_reset"))
        self.anim_speed_label.setText(tr(self.language, "anim_speed_label"))
        self.anim_speed_spin.setSpecialValueText(tr(self.language, "anim_speed_infinite"))
        self.anim_speed_spin.setSuffix(tr(self.language, "anim_speed_suffix"))

    def _update_animation_controls(self):
        has_data = bool(
            self._animation_render_data and (self._animation_render_data.get("paths") or [])
        )
        controls_enabled = has_data and self.animation_enabled
        for w in (
            self.anim_start_btn,
            self.anim_reset_btn,
            self.anim_speed_spin,
            self.anim_btn_half,
            self.anim_btn_double,
        ):
            w.setEnabled(controls_enabled)
        self.anim_container.setVisible(self.animation_enabled)
        self._refresh_animation_texts()

    def _on_anim_speed_changed(self, value: float):
        if value <= 0.0:
            self._animation_speed = math.inf
        else:
            self._animation_speed = value

    def _apply_anim_speed_factor(self, factor: float):
        if factor <= 0:
            return
        current = self.anim_speed_spin.value()
        if current <= 0.0:
            return
        new_val = max(0.25, min(self.anim_speed_spin.maximum(), current * factor))
        self.anim_speed_spin.setValue(new_val)

    def _toggle_animation(self):
        if not self.animation_enabled:
            return
        if not self._animation_render_data:
            return
        max_pts = self._max_animation_points()
        if max_pts <= 1:
            return
        if not self._animation_running:
            if self._animation_progress >= max_pts:
                self._animation_progress = 0.0
            self._animation_last_time = time.monotonic()
            self._animation_running = True
            self._animation_timer.start(16)
            self._update_animation_controls()
            self._render_animation_frame()
        else:
            self._stop_animation()

    def _stop_animation(self):
        self._animation_timer.stop()
        self._animation_running = False
        self._animation_last_time = None
        self._update_animation_controls()

    def _reset_animation_state(self):
        self._stop_animation()
        self._animation_progress = 0.0
        self._update_animation_controls()

    def _reset_animation(self):
        if not self.animation_enabled or not self._animation_render_data:
            return
        self._stop_animation()
        self._animation_progress = 0.0
        self._render_animation_frame()

    def _on_animation_tick(self):
        if not self._animation_render_data:
            self._stop_animation()
            return
        max_pts = self._max_animation_points()
        if max_pts <= 1:
            self._stop_animation()
            return

        now = time.monotonic()
        if self._animation_last_time is None:
            self._animation_last_time = now
        dt = max(0.0, now - self._animation_last_time)
        self._animation_last_time = now

        if math.isinf(self._animation_speed):
            self._animation_progress = max_pts
        else:
            self._animation_progress += self._animation_speed * dt

        if self._animation_progress >= max_pts:
            self._animation_progress = max_pts
            self._render_animation_frame()
            self._stop_animation()
            return

        self._render_animation_frame()

    def _render_animation_frame(self):
        data = self._animation_render_data
        if not data:
            return
        tracks = data.get("tracks") or []
        paths = data.get("paths") or []
        width = data.get("width", self.canvas_width)
        height = data.get("height", self.canvas_height)
        bg = data.get("bg_color", self.bg_color)
        svg_parts = [
            '<?xml version="1.0" standalone="no"?>',
            f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}"',
            '     xmlns="http://www.w3.org/2000/svg" version="1.1">',
            f'  <rect x="0" y="0" width="{width}" height="{height}" fill="{bg}"/>',
        ]

        for track_entry in tracks:
            pts = track_entry.get("points") or []
            if len(pts) < 2:
                continue
            x0, y0 = pts[0]
            cmds = [f"M {x0:.3f} {y0:.3f}"]
            for (x, y) in pts[1:]:
                cmds.append(f"L {x:.3f} {y:.3f}")
            path_d = " ".join(cmds)
            width_stroke = track_entry.get("stroke_width", 1.0)
            svg_parts.append(
                f'  <path d="{path_d}" fill="none" stroke="#808080" stroke-width="{width_stroke}"/>'
            )

        current = self._animation_progress
        for entry in paths:
            pts = entry.get("points") or []
            if math.isinf(current):
                count = len(pts)
            else:
                count = int(min(len(pts), math.floor(current)))
            if count <= 0:
                continue
            t_points = pts[:count]
            x0, y0 = t_points[0]
            cmds = [f"M {x0:.3f} {y0:.3f}"]
            for (x, y) in t_points[1:]:
                cmds.append(f"L {x:.3f} {y:.3f}")
            path_d = " ".join(cmds)
            color = entry.get("color", "#000000")
            width_stroke = entry.get("stroke_width", 1.0)
            svg_parts.append(
                f'  <path d="{path_d}" fill="none" stroke="{color}" stroke-width="{width_stroke}"/>'
            )

        svg_parts.append("</svg>")
        self.load_svg("\n".join(svg_parts))

    # ----- Actions -----

    def open_layer_manager(self):
        dlg = LayerManagerDialog(
            self.layers,
            lang=self.language,
            parent=self,
            points_per_path=self.points_per_path,
        )
        if dlg.exec() == QDialog.Accepted:
            self.layers = dlg.get_layers()
            self.update_svg()

    def open_modular_track_editor(self):
        dlg = ModularTrackEditorDialog(
            lang=self.language,
            parent=self,
        )
        dlg.exec()

    def open_shape_lab(self):
        if getattr(self, "_shape_lab_window", None) is None:
            self._shape_lab_window = ShapeDesignLabWindow(self)
        self._shape_lab_window.show()
        self._shape_lab_window.raise_()
        self._shape_lab_window.activateWindow()

    def edit_bg_color(self):
        # couleur actuelle (texte tel que saisi)
        current = self.bg_color

        # ouvre le color picker complet
        picked = ColorPickerDialog.get_color(
            initial_text=current,
            lang=self.language,
            parent=self,
        )
        if picked is None:
            return  # utilisateur a annulé

        # on vérifie que la couleur est valide
        norm = normalize_color_string(picked)
        if norm is None:
            QMessageBox.warning(
                self,
                tr(self.language, "color_invalid_title"),
                tr(self.language, "color_invalid_text"),
            )
            return

        # on garde le texte saisi (nom, hex ou HSL)
        self.bg_color = picked
        self.update_svg()

    def edit_canvas_settings(self):
        dlg = QDialog(self)
        dlg.setWindowTitle(tr(self.language, "canvas_dialog_title"))
        layout = QFormLayout(dlg)

        spin_w = QSpinBox()
        spin_w.setRange(100, 20000)
        spin_w.setValue(self.canvas_width)

        spin_h = QSpinBox()
        spin_h.setRange(100, 20000)
        spin_h.setValue(self.canvas_height)

        spin_pts = QSpinBox()
        spin_pts.setRange(500, 100000)
        spin_pts.setValue(self.points_per_path)

        layout.addRow(tr(self.language, "canvas_label_width"), spin_w)
        layout.addRow(tr(self.language, "canvas_label_height"), spin_h)
        layout.addRow(tr(self.language, "canvas_label_points"), spin_pts)

        btn_box = QHBoxLayout()
        btn_ok = QPushButton(tr(self.language, "dlg_ok"))
        btn_cancel = QPushButton(tr(self.language, "dlg_cancel"))
        btn_ok.clicked.connect(dlg.accept)
        btn_cancel.clicked.connect(dlg.reject)
        btn_box.addWidget(btn_ok)
        btn_box.addWidget(btn_cancel)
        layout.addRow(btn_box)

        if dlg.exec() == QDialog.Accepted:
            self.canvas_width = spin_w.value()
            self.canvas_height = spin_h.value()
            self.points_per_path = spin_pts.value()
            self.update_svg()

    # ----- Sauvegarde / chargement JSON -----

    def _layers_to_json_struct(self):
        data_layers = []
        for layer in self.layers:
            data_layer = {
                "name": layer.name,
                "enable": layer.enable,
                "zoom": getattr(layer, "zoom", 1.0),
                "translate_x": getattr(layer, "translate_x", 0.0),
                "translate_y": getattr(layer, "translate_y", 0.0),
                "rotate_deg": getattr(layer, "rotate_deg", 0.0),
                "gears": [],
                "paths": [],
            }
            for g in layer.gears:
                data_layer["gears"].append({
                    "name": g.name,
                    "gear_type": g.gear_type,
                    "size": g.size,
                    "outer_size": g.outer_size,
                    "relation": g.relation,
                    "modular_notation": getattr(g, "modular_notation", None),
                    "dsl_expression": getattr(g, "dsl_expression", None),
                })
            for p in layer.paths:
                data_layer["paths"].append({
                    "name": p.name,
                    "enable": p.enable,
                    "hole_offset": p.hole_offset,
                    "phase_offset": p.phase_offset,
                    "color": p.color,  # ce que tu as tapé
                    "color_norm": getattr(p, "color_norm", None),  # peut être None
                    "stroke_width": p.stroke_width,
                    "zoom": getattr(p, "zoom", 1.0),
                    "translate_x": getattr(p, "translate_x", 0.0),
                    "translate_y": getattr(p, "translate_y", 0.0),
                    "rotate_deg": getattr(p, "rotate_deg", 0.0),
                })
            data_layers.append(data_layer)
        return data_layers

    def _layers_from_json_struct(self, data_layers):
        layers: List[LayerConfig] = []
        for ld in data_layers:
            gears = []
            for gd in ld.get("gears", []):
                gears.append(
                    GearConfig(
                        name=gd.get("name", "Engrenage"),
                        gear_type=gd.get("gear_type", "roue"),
                        size=int(gd.get("size", 30)),
                        outer_size=int(gd.get("outer_size", 0)),
                        relation=gd.get("relation", "stationnaire"),
                        modular_notation=gd.get("modular_notation"),
                        dsl_expression=gd.get("dsl_expression"),
                    )
                )
            paths = []
            for pd in ld.get("paths", []):
                color_input = pd.get("color", "#000000")
                color_norm = pd.get("color_norm")
                if color_norm is None:
                    # compat : JSON ancien → on recalcule
                    color_norm = normalize_color_string(color_input) or "#000000"

                paths.append(
                    PathConfig(
                        name=pd.get("name", "Tracé"),
                        enable=bool(pd.get("enable", True)),
                        hole_offset=float(pd.get("hole_offset", pd.get("hole_index", 1.0))),
                        phase_offset=float(pd.get("phase_offset", 0.0)),
                        color=color_input,
                        color_norm=color_norm,
                        stroke_width=float(pd.get("stroke_width", 1.0)),
                        zoom=float(pd.get("zoom", 1.0)),
                        translate_x=float(pd.get("translate_x", 0.0)),
                        translate_y=float(pd.get("translate_y", 0.0)),
                        rotate_deg=float(pd.get("rotate_deg", 0.0)),
                    )
                )
            enable = bool(ld.get("enable", ld.get("visible", True)))
            if not enable:
                for path in paths:
                    path.enable = False
            elif paths and not any(path.enable for path in paths):
                for path in paths:
                    path.enable = True
            layers.append(
                LayerConfig(
                    name=ld.get("name", "Couche"),
                    enable=enable,
                    zoom=float(ld.get("zoom", 1.0)),
                    translate_x=float(ld.get("translate_x", 0.0)),
                    translate_y=float(ld.get("translate_y", 0.0)),
                    rotate_deg=float(ld.get("rotate_deg", 0.0)),
                    gears=gears,
                    paths=paths,
                )
            )
        if layers and not any(layer.enable for layer in layers):
            layers[0].enable = True
            for path in layers[0].paths:
                path.enable = True
        return layers

    def _config_file_path(self) -> str:
        base_dir = QStandardPaths.writableLocation(QStandardPaths.AppConfigLocation)
        if not base_dir:
            base_dir = os.path.expanduser("~")
        try:
            os.makedirs(base_dir, exist_ok=True)
        except Exception:
            pass
        return os.path.join(base_dir, "spirosim_config.json")

    def _gather_state_dict(self, include_window: bool = False):
        data = {
            "version": 1,
            "language": self.language,
            "bg_color": self.bg_color,
            "canvas_width": self.canvas_width,
            "canvas_height": self.canvas_height,
            "points_per_path": self.points_per_path,
            "animation_enabled": self.animation_enabled,
            "animation_speed": self.anim_speed_spin.value(),
            "show_track": self.show_track,
            "layers": self._layers_to_json_struct(),
        }

        if include_window:
            try:
                geom = self.saveGeometry()
                if geom and not geom.isEmpty():
                    data["window_geometry"] = bytes(geom.toBase64()).decode("ascii")
            except Exception:
                pass
            try:
                data["window_state"] = int(self.windowState())
            except Exception:
                pass

        return data

    def _apply_state_dict(self, data, *, apply_window: bool, refresh: bool):
        self.language = localisation.resolve_language(data.get("language", self.language))
        self.bg_color = data.get("bg_color", self.bg_color)
        self.canvas_width = int(data.get("canvas_width", self.canvas_width))
        self.canvas_height = int(data.get("canvas_height", self.canvas_height))
        self.points_per_path = int(data.get("points_per_path", self.points_per_path))

        anim_enabled_val = data.get("animation_enabled")
        if anim_enabled_val is not None:
            self.animation_enabled = bool(anim_enabled_val)
        self._set_animation_enabled(self.animation_enabled)

        saved_speed = data.get("animation_speed")
        if saved_speed is not None:
            try:
                self.anim_speed_spin.setValue(float(saved_speed))
            except Exception:
                pass

        show_track_val = data.get("show_track")
        if show_track_val is not None:
            self.show_track = bool(show_track_val)
            try:
                self.act_show_track.setChecked(self.show_track)
            except Exception:
                pass

        if "layers" in data:
            layers_struct = data.get("layers", [])
            self.layers = self._layers_from_json_struct(layers_struct)

        if apply_window:
            geom_b64 = data.get("window_geometry")
            if geom_b64:
                try:
                    geom_bytes = QByteArray.fromBase64(geom_b64.encode("ascii"))
                    if geom_bytes and not geom_bytes.isEmpty():
                        self.restoreGeometry(geom_bytes)
                        self._geometry_restored = True
                except Exception:
                    pass
            state_val = data.get("window_state")
            if state_val is not None:
                try:
                    self.setWindowState(Qt.WindowState(int(state_val)))
                except Exception:
                    pass

        if refresh:
            self.apply_language()
            self._update_svg_size()
            self.update_svg()

    def _load_persisted_state(self) -> bool:
        cfg_path = self._config_file_path()
        if not os.path.exists(cfg_path):
            return False

        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return False

        self._apply_state_dict(data, apply_window=True, refresh=True)
        return True

    def _save_persisted_state(self):
        cfg_path = self._config_file_path()
        data = self._gather_state_dict(include_window=True)
        try:
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    def save_to_json(self):
        filename, _ = QFileDialog.getSaveFileName(
            self,
            tr(self.language, "menu_file_save_json"),
            "",
            "JSON (*.json)",
        )
        if not filename:
            return

        data = self._gather_state_dict(include_window=False)

        try:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def load_from_json(self):
        filename, _ = QFileDialog.getOpenFileName(
            self,
            tr(self.language, "menu_file_load_json"),
            "",
            "JSON (*.json)",
        )
        if not filename:
            return

        try:
            with open(filename, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            return

        self._apply_state_dict(data, apply_window=False, refresh=True)

    # ----- Export SVG / PNG -----

    def export_svg(self):
        filename, _ = QFileDialog.getSaveFileName(
            self,
            tr(self.language, "menu_file_export_svg"),
            "",
            "SVG (*.svg)",
        )
        if not filename:
            return

        svg_data = layers_to_svg(
            self.layers,
            width=self.canvas_width,
            height=self.canvas_height,
            bg_color=self.bg_color,
            points_per_path=self.points_per_path,
        )

        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(svg_data)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def export_png(self):
        # D'abord demander la taille de sortie
        dlg = QDialog(self)
        dlg.setWindowTitle(tr(self.language, "export_png_dialog_title"))
        layout = QFormLayout(dlg)

        spin_w = QSpinBox()
        spin_w.setRange(100, 20000)
        spin_w.setValue(max(self.canvas_width, 2000))

        spin_h = QSpinBox()
        spin_h.setRange(100, 20000)
        spin_h.setValue(max(self.canvas_height, 2000))

        layout.addRow(tr(self.language, "export_png_width"), spin_w)
        layout.addRow(tr(self.language, "export_png_height"), spin_h)

        btn_box = QHBoxLayout()
        btn_ok = QPushButton(tr(self.language, "dlg_ok"))
        btn_cancel = QPushButton(tr(self.language, "dlg_cancel"))
        btn_ok.clicked.connect(dlg.accept)
        btn_cancel.clicked.connect(dlg.reject)
        btn_box.addWidget(btn_ok)
        btn_box.addWidget(btn_cancel)
        layout.addRow(btn_box)

        if dlg.exec() != QDialog.Accepted:
            return

        out_w = spin_w.value()
        out_h = spin_h.value()

        filename, _ = QFileDialog.getSaveFileName(
            self,
            tr(self.language, "menu_file_export_png"),
            "",
            "PNG (*.png)",
        )
        if not filename:
            return

        # Rendu SVG -> QImage haute résolution
        image = QImage(out_w, out_h, QImage.Format_ARGB32)
        bg_norm = normalize_color_string(self.bg_color) or "#ffffff"
        image.fill(QColor(bg_norm))

        # Générer un SVG aux dimensions souhaitées
        svg_data = layers_to_svg(
            self.layers,
            width=out_w,
            height=out_h,
            bg_color=bg_norm,
            points_per_path=self.points_per_path,
        )

        renderer = QSvgRenderer(QByteArray(svg_data.encode("utf-8")))
        painter = QPainter(image)
        renderer.render(painter)
        painter.end()

        try:
            image.save(filename, "PNG")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def closeEvent(self, event):
        try:
            self._save_persisted_state()
            self._stop_animation()
        finally:
            super().closeEvent(event)

def main():
    app = QApplication(sys.argv)
    window = SpiroWindow()
    if not getattr(window, "_geometry_restored", False):
        window.resize(1000, 1000)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
