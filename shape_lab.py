from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import List, Optional

from PySide6.QtCore import QStandardPaths, QTimer, Qt
from PySide6.QtGui import QColor, QPainterPath, QPen
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QDialog,
    QFormLayout,
    QGraphicsItem,
    QGraphicsLineItem,
    QGraphicsPathItem,
    QGraphicsScene,
    QGraphicsView,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QPlainTextEdit,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

import modular_tracks_2 as modular_tracks
from shape_rsdl import (
    CircleSpec,
    DropSpec,
    RsdlParseError,
    EllipseSpec,
    OblongSpec,
    PolygonSpec,
    RingSpec,
    is_modular_expression,
    normalize_rsdl_text,
    parse_analytic_expression,
)
from shape_geometry import (
    ArcSegment,
    EllipseCurve,
    LineSegment,
    ModularTrackCurve,
    build_circle,
    build_drop,
    build_oblong,
    build_rounded_polygon,
)


@dataclass
class ShapeVariant:
    name: str
    expression: str
    visible: bool = True


def _shape_lab_data_path() -> str:
    base_dir = QStandardPaths.writableLocation(QStandardPaths.AppConfigLocation)
    if not base_dir:
        base_dir = os.path.expanduser("~")
    try:
        os.makedirs(base_dir, exist_ok=True)
    except Exception:
        pass
    return os.path.join(base_dir, "spirosim_rsdl_parts.json")


class ShapeDesignLabWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self._variants: List[ShapeVariant] = []
        self._variant_counter = 1

        splitter = QSplitter(Qt.Horizontal)
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(splitter)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Analytic", "Modular"])
        mode_row.addWidget(self.mode_combo)
        mode_row.addStretch(1)
        left_layout.addLayout(mode_row)

        self.rsdl_editor = QPlainTextEdit()
        self.rsdl_editor.setPlaceholderText("Enter RSDL expression")
        left_layout.addWidget(self.rsdl_editor)

        compile_row = QHBoxLayout()
        self.auto_compile = QCheckBox("Auto-compile")
        self.auto_compile.setChecked(True)
        self.compile_button = QPushButton("Compile")
        compile_row.addWidget(self.auto_compile)
        compile_row.addStretch(1)
        compile_row.addWidget(self.compile_button)
        left_layout.addLayout(compile_row)

        ring_row = QHBoxLayout()
        self.reference_ring_edit = QLineEdit()
        self.reference_ring_edit.setPlaceholderText("R(Ni,No)")
        ring_row.addWidget(QLabel("Reference ring:"))
        ring_row.addWidget(self.reference_ring_edit)
        left_layout.addLayout(ring_row)

        self.diagnostics = QTreeWidget()
        self.diagnostics.setHeaderLabels(["Diagnostics"])
        left_layout.addWidget(self.diagnostics, stretch=1)

        self.quick_params_group = QGroupBox("Quick parameters")
        self.quick_params_layout = QFormLayout(self.quick_params_group)
        left_layout.addWidget(self.quick_params_group)

        variant_group = QGroupBox("Variants")
        variant_layout = QVBoxLayout(variant_group)
        self.variants_table = QTableWidget(0, 3)
        self.variants_table.setHorizontalHeaderLabels(["Name", "Expression", "Visible"])
        self.variants_table.horizontalHeader().setStretchLastSection(True)
        variant_layout.addWidget(self.variants_table)

        variant_buttons = QHBoxLayout()
        self.add_variant_button = QToolButton()
        self.add_variant_button.setText("+")
        self.dup_variant_button = QToolButton()
        self.dup_variant_button.setText("Duplicate")
        self.remove_variant_button = QToolButton()
        self.remove_variant_button.setText("-")
        self.up_variant_button = QToolButton()
        self.up_variant_button.setText("Up")
        self.down_variant_button = QToolButton()
        self.down_variant_button.setText("Down")
        for btn in [
            self.add_variant_button,
            self.dup_variant_button,
            self.remove_variant_button,
            self.up_variant_button,
            self.down_variant_button,
        ]:
            variant_buttons.addWidget(btn)
        variant_layout.addLayout(variant_buttons)
        left_layout.addWidget(variant_group)

        splitter.addWidget(left_panel)

        self.scene = QGraphicsScene()
        self.preview = QGraphicsView(self.scene)
        self.preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        splitter.addWidget(self.preview)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([360, 720])

        self.quality_combo = QComboBox()
        self.quality_combo.addItems(["Draft", "Normal", "High"])
        left_layout.addWidget(self.quality_combo)

        self._debounce_timer = QTimer(self)
        self._debounce_timer.setInterval(350)
        self._debounce_timer.setSingleShot(True)

        self.compile_button.clicked.connect(self.compile_now)
        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)
        self.rsdl_editor.textChanged.connect(self._schedule_compile)
        self.reference_ring_edit.textChanged.connect(self._schedule_compile)
        self._debounce_timer.timeout.connect(self.compile_now)
        self.add_variant_button.clicked.connect(self._add_variant_from_editor)
        self.dup_variant_button.clicked.connect(self._duplicate_variant)
        self.remove_variant_button.clicked.connect(self._remove_variant)
        self.up_variant_button.clicked.connect(lambda: self._move_variant(-1))
        self.down_variant_button.clicked.connect(lambda: self._move_variant(1))
        self.quality_combo.currentTextChanged.connect(self._update_preview)
        self.variants_table.itemChanged.connect(self._on_variant_item_changed)

        self._on_mode_changed(self.mode_combo.currentText())
        self._load_persisted_data()

    def _normalize_editor_text(self) -> str:
        raw = self.rsdl_editor.toPlainText()
        normalized = normalize_rsdl_text(raw)
        if normalized != raw:
            self.rsdl_editor.blockSignals(True)
            self.rsdl_editor.setPlainText(normalized)
            self.rsdl_editor.blockSignals(False)
        ring_raw = self.reference_ring_edit.text()
        ring_normalized = normalize_rsdl_text(ring_raw)
        if ring_normalized != ring_raw:
            self.reference_ring_edit.blockSignals(True)
            self.reference_ring_edit.setText(ring_normalized)
            self.reference_ring_edit.blockSignals(False)
        return normalized

    def set_expression(self, expression: str, *, mode: Optional[str] = None, reference_ring: Optional[str] = None) -> None:
        normalized = normalize_rsdl_text(expression or "")
        if mode is None and normalized:
            mode = "Modular" if is_modular_expression(normalized) else "Analytic"
        if mode in ("Analytic", "Modular"):
            self.mode_combo.setCurrentText(mode)
        if reference_ring is not None:
            self.reference_ring_edit.setText(normalize_rsdl_text(reference_ring))
        self.rsdl_editor.setPlainText(normalized)
        self.compile_now()

    def current_expression(self) -> str:
        return normalize_rsdl_text(self.rsdl_editor.toPlainText())

    def current_expression_valid(self) -> bool:
        expr = normalize_rsdl_text(self.rsdl_editor.toPlainText())
        if not expr:
            return False
        mode = self.mode_combo.currentText()
        if mode == "Analytic":
            try:
                parse_analytic_expression(expr)
            except RsdlParseError:
                return False
            return True
        valid, rest, has_piece = modular_tracks.split_valid_modular_notation(expr)
        return bool(valid) and not rest and has_piece

    def _load_persisted_data(self) -> None:
        data_path = _shape_lab_data_path()
        if not os.path.exists(data_path):
            return
        try:
            with open(data_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception:
            return

        mode = data.get("mode")
        if mode in ("Analytic", "Modular"):
            self.mode_combo.setCurrentText(mode)
        expr = normalize_rsdl_text(data.get("expression", ""))
        self.rsdl_editor.setPlainText(expr)
        ring_expr = normalize_rsdl_text(data.get("reference_ring", ""))
        self.reference_ring_edit.setText(ring_expr)
        quality = data.get("quality")
        if quality in {"Draft", "Normal", "High"}:
            self.quality_combo.setCurrentText(quality)

        variants = []
        for item in data.get("variants", []) or []:
            name = item.get("name", "Variant")
            expression = normalize_rsdl_text(item.get("expression", ""))
            visible = bool(item.get("visible", True))
            variants.append(ShapeVariant(name, expression, visible))
        if variants:
            self._variants = variants
            self._variant_counter = max(self._variant_counter, len(variants) + 1)
            self._refresh_variants_table()
        self.compile_now()

    def save_persisted_data(self) -> None:
        data = {
            "mode": self.mode_combo.currentText(),
            "expression": self.current_expression(),
            "reference_ring": normalize_rsdl_text(self.reference_ring_edit.text()),
            "quality": self.quality_combo.currentText(),
            "variants": [
                {
                    "name": variant.name,
                    "expression": normalize_rsdl_text(variant.expression),
                    "visible": variant.visible,
                }
                for variant in self._variants
            ],
        }
        data_path = _shape_lab_data_path()
        try:
            with open(data_path, "w", encoding="utf-8") as handle:
                json.dump(data, handle, indent=2, ensure_ascii=False)
        except Exception:
            pass

    def _on_mode_changed(self, text: str) -> None:
        self.reference_ring_edit.setVisible(text == "Modular")
        self.compile_now()

    def _schedule_compile(self) -> None:
        if self.auto_compile.isChecked():
            self._debounce_timer.start()

    def compile_now(self) -> None:
        self._set_diagnostics([])
        self._normalize_editor_text()
        self._update_quick_params()
        self._update_preview()

    def _set_diagnostics(self, messages: List[str]) -> None:
        self.diagnostics.clear()
        for message in messages:
            QTreeWidgetItem(self.diagnostics, [message])

    def _add_variant_from_editor(self) -> None:
        expr = normalize_rsdl_text(self.rsdl_editor.toPlainText())
        name = f"Variant {self._variant_counter}"
        self._variant_counter += 1
        self._variants.append(ShapeVariant(name, expr, True))
        self._refresh_variants_table()
        self._update_preview()

    def _duplicate_variant(self) -> None:
        row = self.variants_table.currentRow()
        if row < 0 or row >= len(self._variants):
            return
        variant = self._variants[row]
        self._variants.insert(row + 1, ShapeVariant(f"{variant.name} copy", variant.expression, variant.visible))
        self._refresh_variants_table()
        self._update_preview()

    def _remove_variant(self) -> None:
        row = self.variants_table.currentRow()
        if row < 0 or row >= len(self._variants):
            return
        self._variants.pop(row)
        self._refresh_variants_table()
        self._update_preview()

    def _move_variant(self, delta: int) -> None:
        row = self.variants_table.currentRow()
        if row < 0 or row >= len(self._variants):
            return
        new_row = max(0, min(row + delta, len(self._variants) - 1))
        if new_row == row:
            return
        self._variants[row], self._variants[new_row] = self._variants[new_row], self._variants[row]
        self._refresh_variants_table()
        self.variants_table.setCurrentCell(new_row, 0)
        self._update_preview()

    def _refresh_variants_table(self) -> None:
        self.variants_table.blockSignals(True)
        self.variants_table.setRowCount(len(self._variants))
        for idx, variant in enumerate(self._variants):
            self.variants_table.setItem(idx, 0, QTableWidgetItem(variant.name))
            self.variants_table.setItem(idx, 1, QTableWidgetItem(variant.expression))
            visible_item = QTableWidgetItem("yes" if variant.visible else "no")
            visible_item.setCheckState(Qt.Checked if variant.visible else Qt.Unchecked)
            self.variants_table.setItem(idx, 2, visible_item)
        self.variants_table.blockSignals(False)

    def _on_variant_item_changed(self, item: QTableWidgetItem) -> None:
        row = item.row()
        if row < 0 or row >= len(self._variants):
            return
        variant = self._variants[row]
        if item.column() == 0:
            variant.name = item.text()
        elif item.column() == 1:
            variant.expression = normalize_rsdl_text(item.text())
            self.variants_table.blockSignals(True)
            item.setText(variant.expression)
            self.variants_table.blockSignals(False)
        elif item.column() == 2:
            variant.visible = item.checkState() == Qt.Checked
        self._update_preview()

    def _sample_count(self) -> int:
        selection = self.quality_combo.currentText()
        if selection == "High":
            return 3000
        if selection == "Normal":
            return 1200
        return 600

    def _compile_expression(self, expr: str) -> Optional[List[tuple[float, float]]]:
        expr = normalize_rsdl_text(expr)
        if not expr:
            return None
        mode = self.mode_combo.currentText()
        try:
            if mode == "Analytic":
                spec = parse_analytic_expression(expr)
                if isinstance(spec, CircleSpec):
                    curve = build_circle(spec.perimeter)
                elif isinstance(spec, RingSpec):
                    curve = build_circle(spec.outer)
                elif isinstance(spec, PolygonSpec):
                    curve = build_rounded_polygon(spec.perimeter, spec.sides, spec.side_size, spec.corner_size)
                elif isinstance(spec, DropSpec):
                    curve = build_drop(spec.perimeter, spec.opposite, spec.half, spec.link)
                elif isinstance(spec, OblongSpec):
                    curve = build_oblong(spec.perimeter, spec.cap_size)
                elif isinstance(spec, EllipseSpec):
                    curve = EllipseCurve(spec.perimeter, spec.axis_a, spec.axis_b)
                else:
                    return None
                return self._sample_curve(curve.length, curve.eval)
            ring_expr = normalize_rsdl_text(self.reference_ring_edit.text()) or "R(96,144)"
            ring_spec = parse_analytic_expression(ring_expr)
            if not isinstance(ring_spec, RingSpec):
                raise RsdlParseError("Reference ring must be a ring expression")
            track = modular_tracks.build_track_from_notation(
                expr,
                inner_size=ring_spec.inner,
                outer_size=ring_spec.outer,
                steps_per_unit=3,
            )
            curve = ModularTrackCurve(
                [
                    LineSegment(seg.start, seg.end) if seg.kind == "line" else ArcSegment(
                        seg.center,
                        seg.radius or 0.0,
                        seg.angle_start or 0.0,
                        seg.angle_end or 0.0,
                    )
                    for seg in track.segments
                ],
                closed=False,
            )
            return self._sample_curve(curve.length, curve.eval)
        except RsdlParseError as exc:
            self._set_diagnostics([str(exc)])
            return None

    def _sample_curve(self, length: float, eval_fn) -> List[tuple[float, float]]:
        samples = self._sample_count()
        points = []
        if length == 0:
            return points
        for i in range(samples):
            s = length * i / max(1, samples - 1)
            x, y, _, _ = eval_fn(s)
            points.append((x, y))
        return points

    def _update_preview(self) -> None:
        self.scene.clear()
        palette = [QColor("#ff6b6b"), QColor("#4dabf7"), QColor("#51cf66"), QColor("#ffa94d")]
        variants = [v for v in self._variants if v.visible]
        if not variants:
            expr = normalize_rsdl_text(self.rsdl_editor.toPlainText())
            variants = [ShapeVariant("Current", expr, True)]
        for idx, variant in enumerate(variants):
            points = self._compile_expression(variant.expression)
            if not points:
                continue
            center_x = sum(p[0] for p in points) / len(points)
            center_y = sum(p[1] for p in points) / len(points)
            path = QPainterPath()
            path.moveTo(points[0][0], -points[0][1])
            for x, y in points[1:]:
                path.lineTo(x, -y)
            item = QGraphicsPathItem(path)
            color = palette[idx % len(palette)]
            item.setPen(QPen(color, 0))
            self.scene.addItem(item)
            marker_pen = QPen(QColor("#111111"), 0)
            marker_pen.setCosmetic(True)
            half = 2.5
            h_line = QGraphicsLineItem(center_x - half, -center_y, center_x + half, -center_y)
            v_line = QGraphicsLineItem(center_x, -center_y - half, center_x, -center_y + half)
            for line in (h_line, v_line):
                line.setPen(marker_pen)
                line.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
                self.scene.addItem(line)
        self.scene.setSceneRect(self.scene.itemsBoundingRect())
        self.preview.fitInView(self.scene.itemsBoundingRect(), Qt.KeepAspectRatio)

    def _update_quick_params(self) -> None:
        for i in reversed(range(self.quick_params_layout.count())):
            item = self.quick_params_layout.itemAt(i)
            if item and item.widget():
                item.widget().deleteLater()
        expr = normalize_rsdl_text(self.rsdl_editor.toPlainText())
        if not expr or self.mode_combo.currentText() != "Analytic":
            return
        try:
            spec = parse_analytic_expression(expr)
        except RsdlParseError:
            return
        if isinstance(spec, CircleSpec):
            spin = self._make_spin(spec.perimeter)
            spin.valueChanged.connect(lambda v: self._set_expression(f"C({v:g})"))
            self.quick_params_layout.addRow("N", spin)
        elif isinstance(spec, RingSpec):
            spin_inner = self._make_spin(spec.inner)
            spin_outer = self._make_spin(spec.outer)
            spin_inner.valueChanged.connect(lambda v: self._set_expression(f"R({v:g},{spin_outer.value():g})"))
            spin_outer.valueChanged.connect(lambda v: self._set_expression(f"R({spin_inner.value():g},{v:g})"))
            self.quick_params_layout.addRow("Ni", spin_inner)
            self.quick_params_layout.addRow("No", spin_outer)
        elif isinstance(spec, PolygonSpec):
            spin_sides = self._make_spin(spec.sides, 2, 12, 0)
            spin_t = self._make_spin(spec.perimeter)
            spin_s = self._make_spin(spec.side_size)
            spin_c = self._make_spin(spec.corner_size)
            spin_sides.valueChanged.connect(
                lambda v: self._set_expression(f"P{int(v)}({spin_t.value():g},{spin_s.value():g}/{spin_c.value():g})")
            )
            spin_t.valueChanged.connect(
                lambda v: self._set_expression(f"P{int(spin_sides.value())}({v:g},{spin_s.value():g}/{spin_c.value():g})")
            )
            spin_s.valueChanged.connect(
                lambda v: self._set_expression(f"P{int(spin_sides.value())}({spin_t.value():g},{v:g}/{spin_c.value():g})")
            )
            spin_c.valueChanged.connect(
                lambda v: self._set_expression(f"P{int(spin_sides.value())}({spin_t.value():g},{spin_s.value():g}/{v:g})")
            )
            self.quick_params_layout.addRow("n", spin_sides)
            self.quick_params_layout.addRow("T", spin_t)
            self.quick_params_layout.addRow("S", spin_s)
            self.quick_params_layout.addRow("C", spin_c)
        elif isinstance(spec, DropSpec):
            spin_t = self._make_spin(spec.perimeter)
            spin_o = self._make_spin(spec.opposite)
            spin_h = self._make_spin(spec.half)
            spin_l = self._make_spin(spec.link)
            spin_t.valueChanged.connect(
                lambda v: self._set_expression(f"D({v:g},{spin_o.value():g}/{spin_h.value():g}/{spin_l.value():g})")
            )
            for spin in (spin_o, spin_h, spin_l):
                spin.valueChanged.connect(
                    lambda _v, s=spin_t, o=spin_o, h=spin_h, l=spin_l:
                    self._set_expression(f"D({s.value():g},{o.value():g}/{h.value():g}/{l.value():g})")
                )
            self.quick_params_layout.addRow("T", spin_t)
            self.quick_params_layout.addRow("O", spin_o)
            self.quick_params_layout.addRow("H", spin_h)
            self.quick_params_layout.addRow("L", spin_l)
        elif isinstance(spec, OblongSpec):
            spin_t = self._make_spin(spec.perimeter)
            spin_k = self._make_spin(spec.cap_size)
            spin_t.valueChanged.connect(lambda v: self._set_expression(f"O({v:g},{spin_k.value():g})"))
            spin_k.valueChanged.connect(lambda v: self._set_expression(f"O({spin_t.value():g},{v:g})"))
            self.quick_params_layout.addRow("T", spin_t)
            self.quick_params_layout.addRow("K", spin_k)
        elif isinstance(spec, EllipseSpec):
            spin_t = self._make_spin(spec.perimeter)
            spin_a = self._make_spin(spec.axis_a)
            spin_b = self._make_spin(spec.axis_b)
            spin_t.valueChanged.connect(
                lambda v: self._set_expression(f"L({v:g},{spin_a.value():g}/{spin_b.value():g})")
            )
            for spin in (spin_a, spin_b):
                spin.valueChanged.connect(
                    lambda _v, s=spin_t, a=spin_a, b=spin_b:
                    self._set_expression(f"L({s.value():g},{a.value():g}/{b.value():g})")
                )
            self.quick_params_layout.addRow("T", spin_t)
            self.quick_params_layout.addRow("A", spin_a)
            self.quick_params_layout.addRow("B", spin_b)

    def _make_spin(self, value: float, min_value: float = 0.1, max_value: float = 10000.0, decimals: int = 3) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(min_value, max_value)
        spin.setDecimals(decimals)
        spin.setValue(float(value))
        return spin

    def _set_expression(self, expression: str) -> None:
        self.rsdl_editor.blockSignals(True)
        self.rsdl_editor.setPlainText(expression)
        self.rsdl_editor.blockSignals(False)
        self.compile_now()


class ShapeDesignLabWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Shape Design Lab")
        self.resize(1100, 700)
        self.lab_widget = ShapeDesignLabWidget(self)
        self.setCentralWidget(self.lab_widget)

    def closeEvent(self, event) -> None:
        self.lab_widget.save_persisted_data()
        super().closeEvent(event)


class ShapeDesignLabDialog(QDialog):
    def __init__(
        self,
        parent=None,
        *,
        initial_expression: str = "",
        mode: Optional[str] = None,
        reference_ring: Optional[str] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Shape Design Lab")
        self.resize(1100, 700)

        self.lab_widget = ShapeDesignLabWidget(self)
        self.lab_widget.set_expression(
            initial_expression,
            mode=mode,
            reference_ring=reference_ring,
        )

        layout = QVBoxLayout(self)
        layout.addWidget(self.lab_widget)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch(1)
        self.ok_button = QPushButton("OK")
        self.cancel_button = QPushButton("Cancel")
        btn_layout.addWidget(self.ok_button)
        btn_layout.addWidget(self.cancel_button)
        layout.addLayout(btn_layout)

        self.ok_button.clicked.connect(self._accept)
        self.cancel_button.clicked.connect(self.reject)
        self.lab_widget.rsdl_editor.textChanged.connect(self._sync_ok_enabled)
        self.lab_widget.mode_combo.currentTextChanged.connect(self._sync_ok_enabled)
        self._result_expression = ""
        self._sync_ok_enabled()

    def _sync_ok_enabled(self) -> None:
        self.ok_button.setEnabled(self.lab_widget.current_expression_valid())

    def _accept(self) -> None:
        if not self.lab_widget.current_expression_valid():
            return
        self._result_expression = self.lab_widget.current_expression()
        self.accept()

    def result_expression(self) -> str:
        return self._result_expression

    def closeEvent(self, event) -> None:
        self.lab_widget.save_persisted_data()
        super().closeEvent(event)
