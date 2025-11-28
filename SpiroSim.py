import sys
import math
import copy
import json
import re
import colorsys
import time
import os
from html import escape  # <-- AJOUT ICI
from generated_colors import COLOR_NAME_TO_HEX
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import modular_tracks_2 as modular_tracks

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
    QPen,   # <-- AJOUT ICI
)
from PySide6.QtCore import (
    QByteArray, 
    Qt,
    Signal,
    QPoint,
    QPointF,
    QSize,
    QTimer,
    QStandardPaths,
)
from PySide6.QtSvgWidgets import QSvgWidget
from PySide6.QtSvg import QSvgRenderer   # <-- AJOUTÉ

# ----- Constante : pas par dent (approximation Spirograph) -----
# On suppose que chaque dent sur le cercle de pas correspond à ~0,65 mm
# de longueur d'arc. Le rayon (en mm) = (pas * dents) / (2π).
PITCH_MM_PER_TOOTH = 0.65

def split_valid_modular_notation(text: str) -> Tuple[str, str, bool]:
    """
    Analyse 'mollement' une notation de piste modulaire, en ignorant les espaces
    et en convertissant tout en MAJUSCULES.

    Retourne (valid, rest, has_piece) où :
      - valid : partie comprise (offset + suites de +A, -C, * ...)
      - rest  : le reste (non compris ou incomplet)
      - has_piece : True s'il y a au moins une pièce (+A, -B, etc.)

    Exemples :
      "-18"    -> ("-18", "", False)   # seulement offset, pas de pièce
      "-18-C"  -> ("-18-C", "", True)
      "-19-C+D"-> ("-19-C+D", "", True)
      "-19- X" -> ("-19", "-X", False)
    """
    # 1) on supprime les espaces et on force les MAJUSCULES
    s = "".join(ch.upper() for ch in text if not ch.isspace())
    n = len(s)
    idx = 0
    has_piece = False

    if n == 0:
        return "", "", False

    # 2) offset signé optionnel : (+/-)?digits*
    if idx < n and s[idx] in "+-":
        idx += 1
    start_digits = idx
    while idx < n and s[idx].isdigit():
        idx += 1
    # on ne force pas la présence de chiffres : "-C" est autorisé syntaxiquement
    # mais dans ce cas offset == 0 et les pièces commencent directement après.

    # 3) suite de "*", "+X" ou "-X"
    while idx < n:
        ch = s[idx]
        if ch == "*":
            idx += 1
            continue
        if ch in "+-":
            if idx + 1 < n and s[idx + 1] in modular_tracks.PIECES:
                has_piece = True
                idx += 2
                continue
            else:
                # signe suivi de quelque chose de non reconnu -> on s'arrête
                break
        # caractère inattendu -> on s'arrête ici
        break

    valid = s[:idx]
    rest = s[idx:]
    return valid, rest, has_piece

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

# ---------- 0) Traductions ----------

TRANSLATIONS = {
    "fr": {
        "app_title": "Spiro / Wild Gears - Visionneuse",

        "menu_layers": "Couches",
        "menu_options": "Options",
        "menu_regen": "Régénérer",
        "menu_layers_manage": "Gérer les couches et les tracés…",
        "menu_options_spacing": "Espacement radial des trous / pas des dents…",
        "menu_options_bgcolor": "Couleur de fond…",
        "menu_options_language": "Langue",
        "menu_lang_fr": "Français",
        "menu_lang_en": "English",
        "menu_regen_animation": "Animation",
        "menu_regen_show_track": "Afficher la piste",
        "menu_regen_draw": "Régénérer le dessin",

        "anim_start": "Démarrer l'animation",
        "anim_pause": "Mettre en pause",
        "anim_reset": "Remettre à zéro",
        "anim_speed_label": "Vitesse (points/s) :",
        "anim_speed_infinite": "∞ (instantané)",
        "anim_speed_suffix": " pts/s",

        "dlg_layers_title": "Gérer les couches et les tracés",
        "dlg_layers_col_name": "Nom",
        "dlg_layers_col_type": "Type",
        "dlg_layers_col_details": "Détails",
        "dlg_layers_add_layer": "Ajouter une couche",
        "dlg_layers_add_path": "Ajouter un tracé",
        "dlg_layers_edit": "Éditer",
        "dlg_layers_remove": "Supprimer",
        "dlg_layers_ok": "OK",
        "dlg_layers_cancel": "Annuler",
        "dlg_layers_must_keep_layer_title": "Impossible",
        "dlg_layers_must_keep_layer_text": "Il doit rester au moins une couche.",
        "dlg_layers_remove_last_path_title": "Supprimer le dernier tracé ?",
        "dlg_layers_remove_last_path_text": "Cette couche n'aura plus aucun tracé. Continuer ?",
        "dlg_layers_need_layer_title": "Aucune couche",
        "dlg_layers_need_layer_text": "Sélectionne une couche (ou un tracé d'une couche) avant d'ajouter un tracé.",

        "dlg_layer_edit_title": "Éditer la couche",
        "dlg_layer_name": "Nom de la couche :",
        "dlg_layer_visible": "Visible",
        "dlg_layer_zoom": "Zoom de la couche :",
        "dlg_layer_num_gears": "Nombre d'engrenages (2 ou 3) :",
        "dlg_layer_gear_label": "Engrenage {index}",
        "dlg_layer_gear_name": "Nom :",
        "dlg_layer_gear_type": "Type :",
        "dlg_layer_gear_teeth": "Dents (roue / int. anneau) :",
        "dlg_layer_gear_outer": "Dents ext. (anneau) :",
        "dlg_layer_gear_relation": "Relation :",
        "dlg_layer_gear_modular_notation": "Piste modulaire (notation) :",

        "dlg_ok": "OK",
        "dlg_cancel": "Annuler",

        "dlg_path_edit_title": "Éditer le tracé",
        "dlg_path_name": "Nom du tracé :",
        "dlg_path_hole_index": "Trou (index, float) :",
        "dlg_path_phase": "Décalage (en dents) :",
        "dlg_path_color": "Couleur (nom CSS4 ou #hex) :",
        "dlg_path_width": "Largeur de trait :",
        "dlg_path_zoom": "Zoom du tracé :",

        "spacing_dialog_title": "Espacement radial des trous",
        "spacing_label": "Espacement (en mm) :",
        "teeth_spacing_label": "Pas des dents (mm/dent) :",

        "bgcolor_dialog_title": "Couleur de fond",
        "bgcolor_label": "Couleur de fond (nom CSS4 ou #hex) :",

        "color_invalid_title": "Couleur invalide",
        "color_invalid_text": "La couleur saisie n'est pas une couleur X11/CSS4 ou hexadécimale valide.",

        "tree_type_layer": "Couche",
        "tree_type_path": "Tracé",

        "default_layer_name": "Couche",
        "default_path_name": "Tracé",
        "default_gear_name": "Engrenage {index}",
        "default_ring_name": "Anneau",
        "default_wheel_name": "Roue",

        "menu_file": "Fichier",
        "menu_file_load_json": "Charger paramètres (JSON)…",
        "menu_file_save_json": "Sauvegarder paramètres (JSON)…",
        "menu_file_export_svg": "Exporter en SVG…",
        "menu_file_export_png": "Exporter en PNG haute résolution…",

        "menu_options_canvas": "Taille du canevas et précision…",

        "export_png_dialog_title": "Exporter en PNG",
        "export_png_width": "Largeur (px) :",
        "export_png_height": "Hauteur (px) :",

        "canvas_dialog_title": "Taille du canevas et précision",
        "canvas_label_width": "Largeur du canevas (px) :",
        "canvas_label_height": "Hauteur du canevas (px) :",
        "canvas_label_points": "Points par tracé :",

        "menu_modular_editor": "Éditeur de piste modulaire…",

        "mod_editor_title": "Éditeur de piste modulaire",
        "mod_editor_notation_label": "Notation :",
        "mod_editor_inner_teeth": "Dents intérieures :",
        "mod_editor_outer_teeth": "Dents extérieures :",
        "mod_editor_pitch": "Pas (mm/dent) :",
        "mod_editor_info_no_piece": "Aucune pièce valide dans la notation.",
        "mod_editor_info_error": "Erreur : {error}",
        "mod_editor_info_empty": "Notation valide, mais piste vide.",
        "mod_editor_info_ok": "Longueur ~ {length:.1f} mm, équivalent ~ {teeth:.1f} dents",

        "dlg_close": "Fermer",
    },
    "en": {
        "app_title": "Spiro / Wild Gears - Viewer",

        "menu_layers": "Layers",
        "menu_options": "Options",
        "menu_regen": "Regenerate",
        "menu_layers_manage": "Manage layers and paths…",
        "menu_options_spacing": "Hole radial spacing / tooth pitch…",
        "menu_options_bgcolor": "Background color…",
        "menu_options_language": "Language",
        "menu_lang_fr": "Français",
        "menu_lang_en": "English",
        "menu_regen_animation": "Animation",
        "menu_regen_show_track": "Show track",
        "menu_regen_draw": "Regenerate drawing",

        "anim_start": "Start animation",
        "anim_pause": "Pause",
        "anim_reset": "Reset",
        "anim_speed_label": "Speed (points/s):",
        "anim_speed_infinite": "∞ (instant)",
        "anim_speed_suffix": " pts/s",

        "dlg_layers_title": "Manage layers and paths",
        "dlg_layers_col_name": "Name",
        "dlg_layers_col_type": "Type",
        "dlg_layers_col_details": "Details",
        "dlg_layers_add_layer": "Add layer",
        "dlg_layers_add_path": "Add path",
        "dlg_layers_edit": "Edit",
        "dlg_layers_remove": "Remove",
        "dlg_layers_ok": "OK",
        "dlg_layers_cancel": "Cancel",
        "dlg_layers_must_keep_layer_title": "Impossible",
        "dlg_layers_must_keep_layer_text": "At least one layer must remain.",
        "dlg_layers_remove_last_path_title": "Remove last path?",
        "dlg_layers_remove_last_path_text": "This layer will have no paths left. Continue?",
        "dlg_layers_need_layer_title": "No layer",
        "dlg_layers_need_layer_text": "Select a layer (or one of its paths) before adding a path.",

        "dlg_layer_edit_title": "Edit layer",
        "dlg_layer_name": "Layer name:",
        "dlg_layer_visible": "Visible",
        "dlg_layer_zoom": "Layer zoom:",
        "dlg_layer_num_gears": "Number of gears (2 or 3):",
        "dlg_layer_gear_label": "Gear {index}",
        "dlg_layer_gear_name": "Name:",
        "dlg_layer_gear_type": "Type:",
        "dlg_layer_gear_teeth": "Teeth (wheel / inner ring):",
        "dlg_layer_gear_outer": "Outer teeth (ring):",
        "dlg_layer_gear_relation": "Relation:",
        "dlg_layer_gear_modular_notation": "Modular track (notation):",

        "dlg_ok": "OK",
        "dlg_cancel": "Cancel",

        "dlg_path_edit_title": "Edit path",
        "dlg_path_name": "Path name:",
        "dlg_path_hole_index": "Hole (index, float):",
        "dlg_path_phase": "Offset (in teeth):",
        "dlg_path_color": "Color (CSS4 name or #hex):",
        "dlg_path_width": "Stroke width:",
        "dlg_path_zoom": "Path zoom:",

        "spacing_dialog_title": "Hole radial spacing",
        "spacing_label": "Spacing (mm):",
        "teeth_spacing_label": "Tooth spacing (mm/tooth):",

        "bgcolor_dialog_title": "Background color",
        "bgcolor_label": "Background color (CSS4 name or #hex):",

        "color_invalid_title": "Invalid color",
        "color_invalid_text": "The color you entered is not a valid X11/CSS4 or hexadecimal color.",

        "tree_type_layer": "Layer",
        "tree_type_path": "Path",

        "default_layer_name": "Layer",
        "default_path_name": "Path",
        "default_gear_name": "Gear {index}",
        "default_ring_name": "Ring",
        "default_wheel_name": "Wheel",

        "menu_file": "File",
        "menu_file_load_json": "Load settings (JSON)…",
        "menu_file_save_json": "Save settings (JSON)…",
        "menu_file_export_svg": "Export as SVG…",
        "menu_file_export_png": "Export as high-res PNG…",

        "menu_options_canvas": "Canvas size and precision…",

        "export_png_dialog_title": "Export PNG",
        "export_png_width": "Width (px):",
        "export_png_height": "Height (px):",

        "canvas_dialog_title": "Canvas size and precision",
        "canvas_label_width": "Canvas width (px):",
        "canvas_label_height": "Canvas height (px):",
        "canvas_label_points": "Points per path:",

        "menu_modular_editor": "Modular track editor…",

        "mod_editor_title": "Modular track editor",
        "mod_editor_notation_label": "Notation:",
        "mod_editor_inner_teeth": "Inner teeth:",
        "mod_editor_outer_teeth": "Outer teeth:",
        "mod_editor_pitch": "Pitch (mm/tooth):",
        "mod_editor_info_no_piece": "No valid piece found in notation.",
        "mod_editor_info_error": "Error: {error}",
        "mod_editor_info_empty": "Notation is valid, but track is empty.",
        "mod_editor_info_ok": "Length ~ {length:.1f} mm, equivalent ~ {teeth:.1f} teeth",

        "dlg_close": "Close",
    },
}


def tr(lang: str, key: str) -> str:
    return TRANSLATIONS.get(lang, TRANSLATIONS["fr"]).get(key, key)


# ---------- 1) Modèle de données : engrenages & paths ----------

GEAR_TYPES = [
    "anneau",    # ring
    "roue",      # wheel
    "triangle",
    "carré",
    "barre",
    "croix",
    "oeil",
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
    teeth: int = 96             # roues, dents int. anneau / anneau modulaire
    outer_teeth: int = 144      # anneau : dents ext. (ex: 150/105) / anneau modulaire
    relation: str = "stationnaire"  # stationnaire / dedans / dehors
    modular_notation: Optional[str] = None  # notation de piste si gear_type == "modulaire"


@dataclass
class PathConfig:
    name: str = "Tracé"
    hole_index: float = 1.0
    phase_offset_teeth: float = 0.0
    color: str = "blue"            # chaîne telle que saisie / affichée
    color_norm: Optional[str] = None  # valeur normalisée (#rrggbb) pour le dessin
    stroke_width: float = 1.2
    zoom: float = 1.0


@dataclass
class LayerConfig:
    name: str = "Couche"
    visible: bool = True
    zoom: float = 1.0                         # zoom de la couche
    gears: List[GearConfig] = field(default_factory=list)  # 2 ou 3 engrenages
    paths: List[PathConfig] = field(default_factory=list)


# ---------- 2) GÉOMÉTRIE ----------

def radius_from_teeth(teeth: int, pitch_mm_per_tooth: float = PITCH_MM_PER_TOOTH) -> float:
    """
    Calcule le rayon (en mm) d’un cercle de pas ayant 'teeth' dents,
    en supposant un pas configurable (par défaut 0,65 mm par dent).
    """
    if teeth <= 0:
        return 0.0
    return (pitch_mm_per_tooth * float(teeth)) / (2.0 * math.pi)


def contact_teeth_for_relation(gear: GearConfig, relation: str) -> int:
    """
    Nombre de dents utilisé pour le contact, selon la relation.
    - anneau + 'dedans' : on utilise les dents intérieures (gear.teeth)
    - anneau + 'dehors' : on utilise les dents extérieures (gear.outer_teeth)
    - roue / autres : gear.teeth
    """
    if gear.gear_type == "anneau":
        if relation == "dehors":
            return gear.outer_teeth or gear.teeth
        else:
            return gear.teeth
    return gear.teeth


def contact_radius_for_relation(
    gear: GearConfig, relation: str, pitch_mm_per_tooth: float = PITCH_MM_PER_TOOTH
) -> float:
    """
    Rayon de contact des dents pour un engrenage donné, selon la relation,
    en mm (via radius_from_teeth).
    """
    t = contact_teeth_for_relation(gear, relation)
    return radius_from_teeth(t, pitch_mm_per_tooth=pitch_mm_per_tooth)


def generate_simple_circle_for_index(
    hole_index: float,
    hole_spacing_mm: float,
    steps: int,
):
    """
    Fallback si on n’a pas assez d’engrenages : on simule un cercle
    dont le rayon dépend du trou indexé.
    On prend un rayon "référence" R_tip = 50 mm.
    Trou 0 : rayon = R_tip
    Trou n : R = R_tip - n * hole_spacing_mm  (n peut être float et négatif)
    R est clampé à >= 0.
    """
    R_tip = 50.0
    d = R_tip - hole_index * hole_spacing_mm
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
    hole_spacing_mm: float = 0.65,
    pitch_mm_per_tooth: float = PITCH_MM_PER_TOOTH,
):
    """
    Génère la courbe pour un path donné, en utilisant la configuration
    du layer (engrenages + organisation).

    Convention :
      - Le PREMIER engrenage de la couche (gears[0]) est stationnaire
        et centré en (0, 0).
      - Le DEUXIÈME engrenage (gears[1]) est mobile et porte les trous du path.
      - path.hole_index est un float, peut être négatif.

    Si le premier engrenage est de type "modulaire", il représente une
    piste virtuelle SuperSpirograph, définie par :
      - g0.teeth        => dents intérieures de l’anneau de base
      - g0.outer_teeth  => dents extérieures de l’anneau de base
      - g0.modular_notation => notation de pièce (ex: "-18-C+D+B-...")
    Dans ce cas, on délègue à modular_tracks.generate_track_base_points.
    """

    hole_index = float(path.hole_index)

    # Pas assez d’engrenages : cercle simple + rotation
    if len(layer.gears) < 2:
        base_points = generate_simple_circle_for_index(
            hole_index, hole_spacing_mm, steps
        )
        teeth_moving = 1
        angle_from_teeth = 2.0 * math.pi * (path.phase_offset_teeth / teeth_moving)
        total_angle = math.pi / 2.0 - angle_from_teeth

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

    # Nombre de dents de contact pour chaque engrenage
    T0 = max(1, contact_teeth_for_relation(g0, relation))
    T1 = max(1, contact_teeth_for_relation(g1, relation))

    # --- Cas 1 : piste modulaire comme premier engrenage ---
    if g0.gear_type == "modulaire" and g0.modular_notation:
        inner_teeth = g0.teeth if g0.teeth > 0 else 1
        outer_teeth = g0.outer_teeth if g0.outer_teeth > 0 else inner_teeth

        base_points = modular_tracks.generate_track_base_points(
            notation=g0.modular_notation,
            wheel_teeth=T1,
            hole_index=hole_index,
            hole_spacing_mm=hole_spacing_mm,
            steps=steps,
            relation=relation,
            inner_teeth=inner_teeth,
            outer_teeth=outer_teeth,
            pitch_mm_per_tooth=pitch_mm_per_tooth,
        )

        teeth_moving = T1
        angle_from_teeth = 2.0 * math.pi * (path.phase_offset_teeth / teeth_moving)
        total_angle = math.pi / 2.0 - angle_from_teeth

        cos_a = math.cos(total_angle)
        sin_a = math.sin(total_angle)

        rotated_points = []
        for (x, y) in base_points:
            xr = x * cos_a - y * sin_a
            yr = x * sin_a + y * cos_a
            rotated_points.append((xr, yr))
        return rotated_points

    # --- Cas 2 : comportement standard (anneau / roue ... ) ---

    # Rayons de contact en mm
    R = contact_radius_for_relation(g0, relation, pitch_mm_per_tooth)
    r = contact_radius_for_relation(g1, relation, pitch_mm_per_tooth)

    if R <= 0 or r <= 0:
        base_points = generate_simple_circle_for_index(
            hole_index, hole_spacing_mm, steps
        )
        teeth_moving = max(1, T1)
        angle_from_teeth = 2.0 * math.pi * (path.phase_offset_teeth / teeth_moving)
        total_angle = math.pi / 2.0 - angle_from_teeth

        cos_a = math.cos(total_angle)
        sin_a = math.sin(total_angle)

        rotated_points = []
        for (x, y) in base_points:
            xr = x * cos_a - y * sin_a
            yr = x * sin_a + y * cos_a
            rotated_points.append((xr, yr))
        return rotated_points

    # Distance du stylo au centre de l’engrenage mobile (d, en mm)
    if g1.gear_type == "anneau":
        R_tip_teeth = g1.outer_teeth or g1.teeth
    else:
        R_tip_teeth = g1.teeth

    R_tip = radius_from_teeth(R_tip_teeth, pitch_mm_per_tooth=pitch_mm_per_tooth)
    d = R_tip - hole_index * hole_spacing_mm
    if d < 0:
        d = 0.0

    # Durée t_max pour "fermer" la courbe (basée sur le ratio des dents)
    if T0 >= 1 and T1 >= 1:
        g = math.gcd(int(T0), int(T1))
        # La période correcte dépend du petit engrenage (mobile) :
        t_max = 2.0 * math.pi * (T1 / g)
    else:
        t_max = 20.0 * math.pi

    base_points = []

    for i in range(steps):
        t = t_max * i / (steps - 1)

        if relation == "dedans":
            # Hypotrochoïde : centre mobile à rayon (R - r)
            R_minus_r = R - r
            k = R_minus_r / r
            x = R_minus_r * math.cos(t) + d * math.cos(k * t)
            y = R_minus_r * math.sin(t) - d * math.sin(k * t)

        elif relation == "dehors":
            # Épitrochoïde : centre mobile à rayon (R + r)
            R_plus_r = R + r
            k = R_plus_r / r
            x = R_plus_r * math.cos(t) - d * math.cos(k * t)
            y = R_plus_r * math.sin(t) - d * math.sin(k * t)

        else:
            # Fallback : simple cercle autour de l’origine de rayon d
            x = d * math.cos(t)
            y = d * math.sin(t)

        base_points.append((x, y))

    # Rotation globale selon le décalage en dents
    # On utilise les dents de contact de l’engrenage mobile pour l’offset.
    teeth_moving = max(1, T1)
    angle_from_teeth = 2.0 * math.pi * (path.phase_offset_teeth / teeth_moving)
    # 0 => motif pointant vers le haut (π/2),
    # positif => tourne vers la droite (horaire),
    # négatif => vers la gauche (anti-horaire).
    total_angle = math.pi / 2.0 - angle_from_teeth

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

def paintEvent(self, event):
    painter = QPainter(self)
    try:
        painter.setRenderHint(QPainter.Antialiasing, True)

        w = self.width()
        h = self.height()

        painter.fillRect(self.rect(), self.palette().window())

        if not self.have_track or not self.points:
            painter.setPen(Qt.gray)
            # on pourrait afficher un texte, mais souvent on laisse vide
            return

        # Échelle et transform pour centrer autour de (0,0) et inverser Y
        scale = self._compute_scale(w, h)
        painter.translate(w / 2.0, h / 2.0)
        painter.scale(scale, -scale)

        # Largeur réelle de la piste (en mm)
        inner_r = (self.pitch_mm * float(self.inner_teeth)) / (2.0 * math.pi)
        outer_r = (self.pitch_mm * float(self.outer_teeth)) / (2.0 * math.pi)
        width_mm = max(outer_r - inner_r, self.pitch_mm)
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

        # 3) dents (petits ticks côté "outer")
        L = self.total_length if self.have_track else 0.0
        segments = getattr(self, "segments", [])

        if L > 0.0 and segments and self.pitch_mm > 0:
            pen_teeth = QPen(QColor("#404040"))
            pen_teeth.setWidthF(0)
            painter.setPen(pen_teeth)
            tooth_len = width_mm * 0.4
            num_teeth = max(1, int(L / self.pitch_mm))
            for k in range(num_teeth):
                s = (k + 0.5) * self.pitch_mm
                (x, y), theta, _ = modular_tracks._interpolate_on_segments(
                    s % L, segments
                )
                nx = -math.sin(theta)
                ny = math.cos(theta)
                bx = x + nx * half_w
                by = y + ny * half_w
                tx = bx + nx * tooth_len
                ty = by + ny * tooth_len
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
    finally:
        painter.end()

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
        self.lang = lang
        self.setWindowTitle("Choisir une couleur" if lang == "fr" else "Pick a color")
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
        self.text_edit.setPlaceholderText("Nom, #hex ou (H, S, L)")
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
        scheme_label = QLabel("Harmonie" if lang == "fr" else "Harmony")
        self.scheme_combo = QComboBox()
        self.scheme_combo.addItems([
            "Aucune" if lang == "fr" else "None",
            "Complémentaire" if lang == "fr" else "Complementary",
            "Analogues" if lang == "fr" else "Analogous",
            "Triadique" if lang == "fr" else "Triadic",
            "Tétradique" if lang == "fr" else "Tetradic",
            "Tints & Shades",
        ])
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
        self.search_edit.setPlaceholderText("Search" if lang == "en" else "Recherche")
        btn_clear = QPushButton("Clear")
        btn_clear.clicked.connect(self.search_edit.clear)
        search_layout.addWidget(self.search_edit, 1)
        search_layout.addWidget(btn_clear)

        right_layout.addLayout(search_layout)

        self.list_widget = QListWidget()
        right_layout.addWidget(self.list_widget, 1)

        main_layout.addLayout(right_layout, 1)

        # --- boutons OK / Annuler ---
        btn_layout = QHBoxLayout()
        btn_ok = QPushButton("OK")
        btn_cancel = QPushButton("Cancel" if lang == "en" else "Annuler")
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
        scheme = self.scheme_combo.currentText()
        if self.lang == "en":
            scheme = {
                "None": "Aucune",
                "Complementary": "Complémentaire",
                "Analogous": "Analogues",
                "Triadic": "Triadique",
                "Tetradic": "Tétradique",
            }.get(scheme, scheme)

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

        if scheme.startswith("Aucune"):
            colors = []
        elif scheme.startswith("Complémentaire"):
            add_h_offset(0)
            add_h_offset(180)
        elif scheme.startswith("Analogues"):
            add_h_offset(-30)
            add_h_offset(0)
            add_h_offset(30)
        elif scheme.startswith("Triadique"):
            add_h_offset(0)
            add_h_offset(120)
            add_h_offset(240)
        elif scheme.startswith("Tétradique"):
            add_h_offset(0)
            add_h_offset(90)
            add_h_offset(180)
            add_h_offset(270)
        elif "Tints" in scheme:
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
    hole_spacing_mm: float = 0.65,
    points_per_path: int = 6000,
    pitch_mm_per_tooth: float = PITCH_MM_PER_TOOTH,
    show_tracks: bool = True,
    return_render_data: bool = False,
) -> str:
    """
    Convertit une liste de LayerConfig -> SVG string.
    Chaque layer visible devient un <g>, chaque path un <path>.
    On applique le zoom de la couche et du tracé avant le centrage/scaling global.
    Quand return_render_data=True, renvoie aussi une structure réutilisable pour
    l'animation (points déjà transformés en pixels).
    """
    all_points = []
    rendered_paths = []  # (layer_name, layer_zoom, path_config, points, path_zoom)
    render_paths = []
    render_tracks = []

    for layer in layers:
        if not layer.visible:
            continue
        layer_zoom = getattr(layer, "zoom", 1.0)

        layer_track_points = None
        layer_track_width_mm = None
        if show_tracks:
            if (
                layer.gears
                and layer.gears[0].gear_type == "modulaire"
                and getattr(layer.gears[0], "modular_notation", "")
            ):
                g0 = layer.gears[0]
                inner_teeth = max(1, int(g0.teeth))
                outer_teeth = int(g0.outer_teeth) if g0.outer_teeth else inner_teeth
                outer_teeth = max(outer_teeth, inner_teeth)

                track = modular_tracks.build_track_from_notation(
                    g0.modular_notation,
                    inner_teeth=inner_teeth,
                    outer_teeth=outer_teeth,
                    pitch_mm_per_tooth=pitch_mm_per_tooth,
                    steps_per_tooth=3,
                )
                if track.points:
                    layer_track_points = track.points
                    r_inner = (pitch_mm_per_tooth * float(inner_teeth)) / (2.0 * math.pi)
                    r_outer = (pitch_mm_per_tooth * float(outer_teeth)) / (2.0 * math.pi)
                    width_mm = max(r_outer - r_inner, pitch_mm_per_tooth)
                    layer_track_width_mm = width_mm * layer_zoom
        for path in layer.paths:
            pts = generate_trochoid_points_for_layer_path(
                layer,
                path,
                steps=points_per_path,
                hole_spacing_mm=hole_spacing_mm,
                pitch_mm_per_tooth=pitch_mm_per_tooth,
            )
            if not pts:
                continue
            path_zoom = getattr(path, "zoom", 1.0)
            zoom = layer_zoom * path_zoom
            pts_zoomed = [(x * zoom, y * zoom) for (x, y) in pts]
            rendered_paths.append((layer.name, layer_zoom, path, pts_zoomed, path_zoom))
            all_points.extend(pts_zoomed)

        if show_tracks and layer_track_points:
            track_zoomed = [
                (x * layer_zoom, y * layer_zoom) for (x, y) in layer_track_points
            ]
            render_tracks.append(
                {
                    "layer_name": layer.name,
                    "points": track_zoomed,
                    "stroke_width_mm": layer_track_width_mm,
                }
            )
            all_points.extend(track_zoomed)

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
    cx = (min_x + max_x) / 2.0
    cy = (min_y + max_y) / 2.0

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
        if not layer.visible:
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
      - visible
      - zoom
      - 2 ou 3 engrenages (type, dents, relation)
      - pour un anneau : dents extérieures / intérieures
    """

    def __init__(
        self,
        layer: LayerConfig,
        lang: str = "fr",
        parent=None,
        pitch_mm_per_tooth: float = PITCH_MM_PER_TOOTH,
    ):
        super().__init__(parent)
        self.lang = lang
        self.setWindowTitle(tr(self.lang, "dlg_layer_edit_title"))
        self.layer = layer
        self.pitch_mm_per_tooth = pitch_mm_per_tooth

        layout = QFormLayout(self)

        self.name_edit = QLineEdit(self.layer.name)
        self.visible_check = QCheckBox(tr(self.lang, "dlg_layer_visible"))
        self.visible_check.setChecked(self.layer.visible)

        self.zoom_spin = QDoubleSpinBox()
        self.zoom_spin.setRange(0.01, 100.0)
        self.zoom_spin.setDecimals(3)
        self.zoom_spin.setValue(getattr(self.layer, "zoom", 1.0))

        self.num_gears_spin = QSpinBox()
        self.num_gears_spin.setRange(2, 3)
        current_gears = max(2, min(3, len(self.layer.gears)))
        self.num_gears_spin.setValue(current_gears)

        layout.addRow(tr(self.lang, "dlg_layer_name"), self.name_edit)
        layout.addRow(self.visible_check)
        layout.addRow(tr(self.lang, "dlg_layer_zoom"), self.zoom_spin)
        layout.addRow(tr(self.lang, "dlg_layer_num_gears"), self.num_gears_spin)

        self.gear_widgets = []
        for i in range(3):
            group_label = QLabel(tr(self.lang, "dlg_layer_gear_label").format(index=i + 1))
            gear_name_edit = QLineEdit()
            gear_type_combo = QComboBox()
            gear_type_combo.addItems(GEAR_TYPES)
            teeth_spin = QSpinBox()
            teeth_spin.setRange(1, 10000)
            outer_spin = QSpinBox()
            outer_spin.setRange(0, 20000)
            rel_combo = QComboBox()
            rel_combo.addItems(RELATIONS)

            # Édition de la notation modulaire (visible seulement pour engrenage 1 + type "modulaire")
            modular_edit = QLineEdit()
            modular_label = QLabel(tr(self.lang, "dlg_layer_gear_modular_notation"))
            modular_button = QPushButton("…")
            modular_button.setFixedWidth(28)

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
            label_teeth = QLabel(tr(self.lang, "dlg_layer_gear_teeth"))
            row3.addWidget(label_teeth)
            row3.addWidget(teeth_spin)
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

            layout.addRow(sub)

            # Restreindre "modulaire" aux engrenages d'indice 0 uniquement
            if i > 0:
                idx_mod = gear_type_combo.findText("modulaire")
                if idx_mod >= 0:
                    gear_type_combo.removeItem(idx_mod)

            gw = dict(
                index=i,
                name_edit=gear_name_edit,
                type_combo=gear_type_combo,
                teeth_spin=teeth_spin,
                outer_spin=outer_spin,
                outer_label=label_outer,
                rel_combo=rel_combo,
                modular_label=modular_label,
                modular_edit=modular_edit,
                modular_button=modular_button,
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
                        teeth=105,
                        outer_teeth=150,
                        relation="stationnaire",
                    )
                else:
                    g = GearConfig(
                        name=tr(self.lang, "default_wheel_name"),
                        gear_type="roue",
                        teeth=30,
                        relation="dedans",
                    )

            gw["name_edit"].setText(g.name)
            # Si jamais un fichier ancien contient "modulaire" sur un engrenage > 0, on le ramène à "roue"
            gear_type = g.gear_type
            if idx > 0 and gear_type == "modulaire":
                gear_type = "roue"

            try:
                type_index = GEAR_TYPES.index(gear_type)
            except ValueError:
                type_index = 0
            gw["type_combo"].setCurrentIndex(type_index)

            gw["teeth_spin"].setValue(g.teeth)
            gw["outer_spin"].setValue(g.outer_teeth if g.outer_teeth > 0 else 0)
            gw["modular_edit"].setText(getattr(g, "modular_notation", "") or "")

            try:
                rel_index = RELATIONS.index(g.relation)
            except ValueError:
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
        t = gw["type_combo"].currentText()
        idx = gw.get("index", 0)

        # "anneau" et "modulaire" ont des dents intérieures / extérieures
        is_ring_like = (t == "anneau" or t == "modulaire")
        gw["outer_label"].setVisible(is_ring_like)
        gw["outer_spin"].setVisible(is_ring_like)

        # La notation modulaire n’est visible que pour l’engrenage 1 et le type "modulaire"
        is_modular = (t == "modulaire" and idx == 0)
        gw["modular_label"].setVisible(is_modular)
        gw["modular_edit"].setVisible(is_modular)
        gw["modular_button"].setVisible(is_modular)

    def _open_modular_editor_from_widget(self, gw: dict):
        if gw.get("index", 0) != 0:
            return
        if gw["type_combo"].currentText() != "modulaire":
            return

        notation = gw["modular_edit"].text().strip()
        inner_teeth = gw["teeth_spin"].value()
        outer_teeth = gw["outer_spin"].value() or inner_teeth

        dlg = ModularTrackEditorDialog(
            lang=self.lang,
            parent=self,
            initial_notation=notation,
            inner_teeth=inner_teeth,
            outer_teeth=outer_teeth,
            pitch_mm_per_tooth=self.pitch_mm_per_tooth,
        )
        if dlg.exec() == QDialog.Accepted:
            gw["modular_edit"].setText(dlg.result_notation())

    def accept(self):
        self.layer.name = self.name_edit.text().strip() or tr(self.lang, "default_layer_name")
        self.layer.visible = self.visible_check.isChecked()
        self.layer.zoom = self.zoom_spin.value()
        num_gears = self.num_gears_spin.value()

        new_gears: List[GearConfig] = []
        for i in range(num_gears):
            gw = self.gear_widgets[i]
            name = gw["name_edit"].text().strip() or tr(self.lang, "default_gear_name").format(index=i + 1)
            gear_type = gw["type_combo"].currentText()

            # Sécurité : on n’autorise "modulaire" que pour le premier engrenage
            if i > 0 and gear_type == "modulaire":
                gear_type = "roue"

            teeth = gw["teeth_spin"].value()
            outer_teeth = gw["outer_spin"].value() if gear_type in ("anneau", "modulaire") else 0

            rel = gw["rel_combo"].currentText()
            if i == 0:
                rel = "stationnaire"

            modular_notation = None
            if gear_type == "modulaire":
                txt = gw["modular_edit"].text().strip()
                if txt:
                    modular_notation = txt

            new_gears.append(
                GearConfig(
                    name=name,
                    gear_type=gear_type,
                    teeth=teeth,
                    outer_teeth=outer_teeth,
                    relation=rel,
                    modular_notation=modular_notation,
                )
            )
        self.layer.gears = new_gears

        super().accept()


class PathEditDialog(QDialog):
    """
    Path :
      - hole_index (float, positif ou négatif)
      - décalage en dents
      - couleur (CSS4 ou hex) avec validation X11/CSS4/hex
      - largeur de trait
      - zoom
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
        self.hole_spin.setValue(self.path.hole_index)

        self.phase_spin = QDoubleSpinBox()
        self.phase_spin.setRange(-1000.0, 1000.0)
        self.phase_spin.setDecimals(3)
        self.phase_spin.setValue(self.path.phase_offset_teeth)

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

        layout.addRow(tr(self.lang, "dlg_path_name"), self.name_edit)
        layout.addRow(tr(self.lang, "dlg_path_hole_index"), self.hole_spin)
        layout.addRow(tr(self.lang, "dlg_path_phase"), self.phase_spin)
        layout.addRow(tr(self.lang, "dlg_path_color"), color_row)
        layout.addRow(tr(self.lang, "dlg_path_width"), self.stroke_spin)
        layout.addRow(tr(self.lang, "dlg_path_zoom"), self.zoom_spin)

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
        self.path.hole_index = self.hole_spin.value()
        self.path.phase_offset_teeth = self.phase_spin.value()

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
        super().accept()


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
        pitch_mm_per_tooth: float = PITCH_MM_PER_TOOTH,
    ):
        super().__init__(parent)
        self.lang = lang
        self.setWindowTitle(tr(self.lang, "dlg_layers_title"))
        self.resize(550, 500)

        self.layers: List[LayerConfig] = copy.deepcopy(layers)
        self.pitch_mm_per_tooth: float = pitch_mm_per_tooth

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
        self.btn_remove = QPushButton(tr(self.lang, "dlg_layers_remove"))
        btn_layout.addWidget(self.btn_add_layer)
        btn_layout.addWidget(self.btn_add_path)
        btn_layout.addWidget(self.btn_edit)
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
        self.btn_remove.clicked.connect(self.on_remove)
        self.btn_ok.clicked.connect(self.accept)
        self.btn_cancel.clicked.connect(self.reject)

        self.tree.currentItemChanged.connect(self.on_selection_changed)
        self.tree.itemDoubleClicked.connect(self.on_item_double_clicked)

        self.refresh_tree()

    # --- utilitaires ---

    def _layer_summary(self, layer: LayerConfig) -> str:
        parts = []
        parts.append(f"{len(layer.gears)} engr., zoom {layer.zoom:g}")
        gear_descs = []
        for i, g in enumerate(layer.gears):
            if i == 0:
                type_name = g.gear_type.capitalize()
            else:
                type_name = g.gear_type.lower()

            if g.gear_type in ("anneau", "modulaire") and g.outer_teeth > 0:
                tooth_str = f"{g.outer_teeth}/{g.teeth}"
            else:
                tooth_str = f"{g.teeth}"

            rel = "stat" if g.relation == "stationnaire" else g.relation
            gear_descs.append(f"{type_name} {tooth_str} {rel}")
        if gear_descs:
            parts.append(", ".join(gear_descs))

        # Si le premier engrenage est modulaire, ajouter la notation
        if layer.gears and layer.gears[0].gear_type == "modulaire":
            notation = getattr(layer.gears[0], "modular_notation", None)
            if notation:
                parts.append(f"mod: {notation}")

        return ", ".join(parts)

    def _path_summary(self, path: PathConfig) -> str:
        return f"{path.hole_index:g}, {path.phase_offset_teeth:g}, {path.color}, {path.stroke_width:g}, zoom {path.zoom:g}"

    def refresh_tree(self):
        self.tree.clear()
        current_item_to_select = None

        for li, layer in enumerate(self.layers):
            layer_item = QTreeWidgetItem(
                [layer.name, tr(self.lang, "tree_type_layer"), self._layer_summary(layer)]
            )
            layer_item.setData(0, Qt.UserRole, layer)
            self.tree.addTopLevelItem(layer_item)

            if li == self.selected_layer_idx and self.selected_path_idx is None:
                current_item_to_select = layer_item

            for pi, path in enumerate(layer.paths):
                path_item = QTreeWidgetItem(
                    [path.name, tr(self.lang, "tree_type_path"), self._path_summary(path)]
                )
                path_item.setData(0, Qt.UserRole, path)
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

    def on_item_double_clicked(self, item: QTreeWidgetItem, column: int):
        self.on_edit()

    # --- callbacks boutons ---

    def on_add_layer(self):
        g0 = GearConfig(
            name=tr(self.lang, "default_ring_name"),
            gear_type="anneau",
            teeth=105,
            outer_teeth=150,
            relation="stationnaire",
        )
        g1 = GearConfig(
            name=tr(self.lang, "default_wheel_name"),
            gear_type="roue",
            teeth=30,
            relation="dedans",
        )
        new_layer = LayerConfig(
            name=f"{tr(self.lang, 'default_layer_name')} {len(self.layers) + 1}",
            visible=True,
            zoom=1.0,
            gears=[g0, g1],
            paths=[PathConfig(name=f"{tr(self.lang, 'default_path_name')} 1", hole_index=1.0, zoom=1.0)],
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
            hole_index=1.0,
            zoom=1.0,
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
                pitch_mm_per_tooth=self.pitch_mm_per_tooth,
            )
        else:
            dlg = PathEditDialog(obj, lang=self.lang, parent=self)
        if dlg.exec() == QDialog.Accepted:
            self.refresh_tree()

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

        self.refresh_tree()

    def get_layers(self) -> List[LayerConfig]:
        return self.layers

class ModularTrackView(QWidget):
    """
    Widget de visualisation d'une piste modulaire :
      - polyline centrale (modular_tracks)
      - bande de largeur réelle (inner/outer)
      - dents approximatives le long de la piste
      - ligne rouge perpendiculaire à la tangente de fin
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.points: List[Tuple[float, float]] = []
        self.have_track = False
        self.inner_teeth = 96
        self.outer_teeth = 144
        self.pitch_mm = 0.65
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
        inner_teeth: int,
        outer_teeth: int,
        pitch_mm: float,
    ):
        self.points = track.points or []
        self.segments = track.segments or []
        self.total_length = track.total_length
        self.inner_teeth = max(1, inner_teeth)
        self.outer_teeth = max(self.inner_teeth + 1, outer_teeth)
        self.pitch_mm = pitch_mm
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

        # Largeur réelle de la piste (en mm)
        inner_r = (self.pitch_mm * float(self.inner_teeth)) / (2.0 * math.pi)
        outer_r = (self.pitch_mm * float(self.outer_teeth)) / (2.0 * math.pi)
        width_mm = max(outer_r - inner_r, self.pitch_mm)
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

        # 3) dents (petits ticks côté "outer")
        L = self.total_length if self.have_track else 0.0
        if L > 0.0 and self.segments and self.pitch_mm > 0:
            pen_teeth = QPen(QColor("#404040"))
            pen_teeth.setWidthF(0)
            painter.setPen(pen_teeth)
            tooth_len = width_mm * 0.4
            num_teeth = max(1, int(L / self.pitch_mm))
            for k in range(num_teeth):
                s = (k + 0.5) * self.pitch_mm
                (x, y), theta, _ = modular_tracks._interpolate_on_segments(
                    s % L, self.segments
                )
                nx = -math.sin(theta)
                ny = math.cos(theta)
                bx = x + nx * half_w
                by = y + ny * half_w
                tx = bx + nx * tooth_len
                ty = by + ny * tooth_len
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
      - paramètres inner/outer/pitch
      - vue centrée sur le barycentre (via modular_tracks)
    """

    def __init__(
        self,
        lang: str = "fr",
        parent=None,
        initial_notation: str = "",
        inner_teeth: int = 96,
        outer_teeth: int = 144,
        pitch_mm_per_tooth: float = PITCH_MM_PER_TOOTH,
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
        self.inner_spin.setValue(max(1, inner_teeth))

        self.outer_spin = QSpinBox()
        self.outer_spin.setRange(1, 4000)
        self.outer_spin.setValue(max(1, outer_teeth))

        self.pitch_spin = QDoubleSpinBox()
        self.pitch_spin.setRange(0.01, 5.0)
        self.pitch_spin.setDecimals(3)
        self.pitch_spin.setSingleStep(0.01)
        self.pitch_spin.setValue(max(0.01, pitch_mm_per_tooth))

        params_layout.addWidget(QLabel(tr(self.lang, "mod_editor_inner_teeth")))
        params_layout.addWidget(self.inner_spin)
        params_layout.addWidget(QLabel(tr(self.lang, "mod_editor_outer_teeth")))
        params_layout.addWidget(self.outer_spin)
        params_layout.addWidget(QLabel(tr(self.lang, "mod_editor_pitch")))
        params_layout.addWidget(self.pitch_spin)
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
        self.pitch_spin.valueChanged.connect(self.update_track)

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

        inner_teeth = self.inner_spin.value()
        outer_teeth = self.outer_spin.value()
        pitch = self.pitch_spin.value()

        # Rien à dessiner si aucune pièce
        if not has_piece or not valid:
            self.track_view.clear_track()
            self.info_label.setText(tr(self.lang, "mod_editor_info_no_piece"))
            return

        try:
            track = modular_tracks.build_track_from_notation(
                valid,
                inner_teeth=inner_teeth,
                outer_teeth=outer_teeth,
                pitch_mm_per_tooth=pitch,
                steps_per_tooth=3,
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

        self.track_view.set_track(track, inner_teeth, outer_teeth, pitch)
        self.info_label.setText(
            tr(self.lang, "mod_editor_info_ok").format(
                length=track.total_length,
                teeth=track.total_teeth,
            )
        )

# ---------- 7) Fenêtre principale ----------

class SpiroWindow(QWidget):
    def __init__(self):
        super().__init__()

        # Langue par défaut : français
        self.language = "fr"

        # Indicateur de restauration de géométrie
        self._geometry_restored = False

        # Espacement radial des trous en mm
        self.hole_spacing_mm: float = 0.65

        # Espacement des dents en mm/dent
        self.pitch_mm_per_tooth: float = PITCH_MM_PER_TOOTH

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

        self.menu_file.addAction(self.act_load_json)
        self.menu_file.addAction(self.act_save_json)
        self.menu_file.addSeparator()
        self.menu_file.addAction(self.act_export_svg)
        self.menu_file.addAction(self.act_export_png)

        self.act_load_json.triggered.connect(self.load_from_json)
        self.act_save_json.triggered.connect(self.save_to_json)
        self.act_export_svg.triggered.connect(self.export_svg)
        self.act_export_png.triggered.connect(self.export_png)

        # Menu Couches
        self.menu_layers = QMenu(menubar)
        menubar.addMenu(self.menu_layers)
        self.act_manage_layers = QAction(menubar)
        self.menu_layers.addAction(self.act_manage_layers)
        self.act_manage_layers.triggered.connect(self.open_layer_manager)

        # Menu Options
        self.menu_options = QMenu(menubar)
        menubar.addMenu(self.menu_options)
        self.act_spacing = QAction(menubar)
        self.act_spacing.triggered.connect(self.edit_hole_spacing)
        self.menu_options.addAction(self.act_spacing)

        self.act_bg_color = QAction(menubar)
        self.act_bg_color.triggered.connect(self.edit_bg_color)
        self.menu_options.addAction(self.act_bg_color)

        # NOUVELLE OPTION : taille du canevas et précision
        self.act_canvas = QAction(menubar)
        self.act_canvas.triggered.connect(self.edit_canvas_settings)
        self.menu_options.addAction(self.act_canvas)

        # Sous-menu Langue
        self.menu_lang = QMenu(menubar)
        self.menu_options.addMenu(self.menu_lang)

        self.act_lang_fr = QAction(menubar)
        self.act_lang_fr.setCheckable(True)
        self.act_lang_en = QAction(menubar)
        self.act_lang_en.setCheckable(True)
        self.menu_lang.addAction(self.act_lang_fr)
        self.menu_lang.addAction(self.act_lang_en)

        self.act_lang_fr.triggered.connect(lambda: self.set_language("fr"))
        self.act_lang_en.triggered.connect(lambda: self.set_language("en"))

        act_modular_editor = QAction(
            tr(self.language, "menu_modular_editor"), self,
        )
        act_modular_editor.triggered.connect(self.open_modular_track_editor)
        self.menu_options.addAction(act_modular_editor)

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
            teeth=105,        # intérieur
            outer_teeth=150,  # extérieur
            relation="stationnaire",
        )
        g1 = GearConfig(
            name=tr(self.language, "default_wheel_name"),
            gear_type="roue",
            teeth=30,
            relation="dedans",
        )
        base_layer = LayerConfig(
            name=f"{tr(self.language, 'default_layer_name')} 1",
            visible=True,
            zoom=1.0,
            gears=[g0, g1],
            paths=[
                PathConfig(
                    name=f"{tr(self.language, 'default_path_name')} 1",
                    hole_index=1.0,
                    phase_offset_teeth=0.0,
                    color="red",
                    stroke_width=1.2,
                    zoom=1.0,
                ),
                PathConfig(
                    name=f"{tr(self.language, 'default_path_name')} 2",
                    hole_index=2.0,
                    phase_offset_teeth=5.0,
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
        if lang not in TRANSLATIONS:
            lang = "fr"
        self.language = lang
        self.apply_language()

    def apply_language(self):
        self.setWindowTitle(tr(self.language, "app_title"))

        # Menus
        self.menu_file.setTitle(tr(self.language, "menu_file"))
        self.menu_layers.setTitle(tr(self.language, "menu_layers"))
        self.menu_options.setTitle(tr(self.language, "menu_options"))
        self.menu_regen.setTitle(tr(self.language, "menu_regen"))

        # Actions Fichier
        self.act_load_json.setText(tr(self.language, "menu_file_load_json"))
        self.act_save_json.setText(tr(self.language, "menu_file_save_json"))
        self.act_export_svg.setText(tr(self.language, "menu_file_export_svg"))
        self.act_export_png.setText(tr(self.language, "menu_file_export_png"))

        # Actions Options
        self.act_manage_layers.setText(tr(self.language, "menu_layers_manage"))
        self.act_spacing.setText(tr(self.language, "menu_options_spacing"))
        self.act_bg_color.setText(tr(self.language, "menu_options_bgcolor"))
        self.act_canvas.setText(tr(self.language, "menu_options_canvas"))
        self.menu_lang.setTitle(tr(self.language, "menu_options_language"))
        self.act_lang_fr.setText(tr(self.language, "menu_lang_fr"))
        self.act_lang_en.setText(tr(self.language, "menu_lang_en"))
        self.act_animation_enabled.setText(tr(self.language, "menu_regen_animation"))
        self.act_show_track.setText(tr(self.language, "menu_regen_show_track"))
        self.act_regen.setText(tr(self.language, "menu_regen_draw"))

        self._refresh_animation_texts()

        # Checkmarks langue
        self.act_lang_fr.setChecked(self.language == "fr")
        self.act_lang_en.setChecked(self.language == "en")

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
            hole_spacing_mm=self.hole_spacing_mm,
            points_per_path=self.points_per_path,
            pitch_mm_per_tooth=self.pitch_mm_per_tooth,
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
            pitch_mm_per_tooth=self.pitch_mm_per_tooth,
        )
        if dlg.exec() == QDialog.Accepted:
            self.layers = dlg.get_layers()
            self.update_svg()

    def edit_hole_spacing(self):
        dlg = QDialog(self)
        dlg.setWindowTitle(tr(self.language, "spacing_dialog_title"))
        layout = QFormLayout(dlg)
        spin = QDoubleSpinBox()
        spin.setRange(0.01, 10.0)
        spin.setDecimals(3)
        spin.setValue(self.hole_spacing_mm)
        layout.addRow(tr(self.language, "spacing_label"), spin)

        pitch_spin = QDoubleSpinBox()
        pitch_spin.setRange(0.01, 5.0)
        pitch_spin.setDecimals(3)
        pitch_spin.setSingleStep(0.01)
        pitch_spin.setValue(self.pitch_mm_per_tooth)
        layout.addRow(tr(self.language, "teeth_spacing_label"), pitch_spin)

        btn_box = QHBoxLayout()
        btn_ok = QPushButton(tr(self.language, "dlg_ok"))
        btn_cancel = QPushButton(tr(self.language, "dlg_cancel"))
        btn_ok.clicked.connect(dlg.accept)
        btn_cancel.clicked.connect(dlg.reject)
        btn_box.addWidget(btn_ok)
        btn_box.addWidget(btn_cancel)
        layout.addRow(btn_box)

        if dlg.exec() == QDialog.Accepted:
            self.hole_spacing_mm = spin.value()
            self.pitch_mm_per_tooth = pitch_spin.value()
            self.update_svg()

    def open_modular_track_editor(self):
        dlg = ModularTrackEditorDialog(
            lang=self.language,
            parent=self,
            pitch_mm_per_tooth=self.pitch_mm_per_tooth,
        )
        dlg.exec()

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
                "visible": layer.visible,
                "zoom": getattr(layer, "zoom", 1.0),
                "gears": [],
                "paths": [],
            }
            for g in layer.gears:
                data_layer["gears"].append({
                    "name": g.name,
                    "gear_type": g.gear_type,
                    "teeth": g.teeth,
                    "outer_teeth": g.outer_teeth,
                    "relation": g.relation,
                    "modular_notation": getattr(g, "modular_notation", None),
                })
            for p in layer.paths:
                data_layer["paths"].append({
                    "name": p.name,
                    "hole_index": p.hole_index,
                    "phase_offset_teeth": p.phase_offset_teeth,
                    "color": p.color,  # ce que tu as tapé
                    "color_norm": getattr(p, "color_norm", None),  # peut être None
                    "stroke_width": p.stroke_width,
                    "zoom": getattr(p, "zoom", 1.0),
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
                        teeth=int(gd.get("teeth", 30)),
                        outer_teeth=int(gd.get("outer_teeth", 0)),
                        relation=gd.get("relation", "stationnaire"),
                        modular_notation=gd.get("modular_notation"),
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
                        hole_index=float(pd.get("hole_index", 1.0)),
                        phase_offset_teeth=float(pd.get("phase_offset_teeth", 0.0)),
                        color=color_input,
                        color_norm=color_norm,
                        stroke_width=float(pd.get("stroke_width", 1.0)),
                        zoom=float(pd.get("zoom", 1.0)),
                    )
                )
            layers.append(
                LayerConfig(
                    name=ld.get("name", "Couche"),
                    visible=bool(ld.get("visible", True)),
                    zoom=float(ld.get("zoom", 1.0)),
                    gears=gears,
                    paths=paths,
                )
            )
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
            "hole_spacing_mm": self.hole_spacing_mm,
            "pitch_mm_per_tooth": self.pitch_mm_per_tooth,
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
        self.language = data.get("language", self.language)
        self.hole_spacing_mm = float(data.get("hole_spacing_mm", self.hole_spacing_mm))
        self.pitch_mm_per_tooth = float(
            data.get("pitch_mm_per_tooth", self.pitch_mm_per_tooth)
        )
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
            hole_spacing_mm=self.hole_spacing_mm,
            points_per_path=self.points_per_path,
            pitch_mm_per_tooth=self.pitch_mm_per_tooth,
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
            hole_spacing_mm=self.hole_spacing_mm,
            points_per_path=self.points_per_path,
            pitch_mm_per_tooth=self.pitch_mm_per_tooth,
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
