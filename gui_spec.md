# Shape Design Lab – GUI Specification (PySide6)

## 1. Integration Approach

The Shape Design Lab is integrated into the existing PySide6 application as:

- a `QDockWidget` (preferred), or
- a standalone tool `QMainWindow`

It must integrate with:
- the existing rendering infrastructure (2D preview canvas)
- the existing layer/asset model
- the rolling simulation pipeline (via exported assets)

The Lab should be **non-destructive**: it adds capabilities without requiring a refactor of the existing solver or layer UI.

---

## 2. High-Level Layout

Use a horizontal splitter:

- Left: controls and diagnostics
- Right: preview canvas

```text
+--------------------------------------------------+
| Shape Design Lab                                  |
+----------------------+---------------------------+
| Controls (Left)      | Preview (Right)           |
+----------------------+---------------------------+
```

---

## 3. Left Panel – Controls

### 3.1 Mode Selection
Widgets:
- `QComboBox`: `Analytic` / `Modular`
- Changing mode updates visible inputs and compiler pipeline.

### 3.2 RSDL Editor
Widgets:
- `QPlainTextEdit` for the main RSDL expression
- `QCheckBox`: Auto-compile
- `QPushButton`: Compile

Notes:
- RSDL stands for **rolling surface definition language**.
- Canonical RSDL uses uppercase identifiers.

Debounce:
- Use `QTimer` to debounce edits when Auto-compile is enabled (e.g., 250–400 ms).
- Compilation emits results via signals to update preview and diagnostics.

### 3.3 Reference Ring Input (Modular Mode Only)
Widgets:
- `QLineEdit` for `R(Ni,No)`
- Visible and required only in Modular mode.

### 3.4 Diagnostics Panel
Widgets:
- `QTreeView` (preferred) or read-only `QPlainTextEdit`

Shows:
- syntax errors (with span/caret information)
- semantic errors
- warnings
- computed metrics (T, closure error, α/β for P<n>, LUT residual for ellipses)

Severity levels:
- info / warning / error

### 3.5 Quick Parameter Panel (Template-Aware)
Widgets:
- dynamic `QGroupBox` populated when the AST matches known templates.

Controls mapping:

| RSDL Node | Controls |
|---|---|
| `P<n>` | n, T, S, C |
| `d` | T, O, H, L |
| `O` | T, K |
| `L` | T, A, B |
| `C` | N |
| `R` | Ni, No |

Round-trip requirement:
- UI edits update AST
- AST prints canonical RSDL back into the editor

---

## 4. Variant Manager (Overlay Comparison)

Purpose:
- compare multiple expressions at once (e.g., `P2(64,90/8)` vs `P2(60,83/8)`)
- toggle visibility, duplicate, reorder

Widgets:
- `QTableView` or `QListWidget` with:
  - name
  - expression
  - visible checkbox

Actions:
- Add variant (from current editor)
- Duplicate selected
- Remove
- Move up/down

Preview draws all visible variants concurrently.

---

## 5. Preview Canvas (QGraphicsView/QGraphicsScene)

Recommended approach:
- Use `QGraphicsView` with a `QGraphicsScene`.
- Render each curve as a `QGraphicsPathItem` built from sampled points.

Features:
- pan (middle-mouse drag) and zoom (mouse wheel)
- fit-to-view
- optional overlays:
  - start point marker
  - direction arrow
  - segment boundary ticks

Quality presets:
- Draft / Normal / High
- Controls sampling density used to generate polyline points

Coordinate handling:
- Default to translating curves to their centroid for consistent viewing.
- Keep a stable convention for start point visualization.

---

## 6. Modular Track Preview

### 6.1 Phase 1: Graph View
Render:
- connectors as nodes
- segments as edges
- highlight current open branch index
- show list of open branches in a side panel or under diagnostics

### 6.2 Phase 2: Route Preview
Add support for:
- selecting a traversal rule through `I<n>` intersections
- generating a single `Curve2D` route for preview and export

---

## 7. Export Workflow and Main-App Integration

Export intent:
- add a designed shape into the main app as a reusable asset and/or new layer input

Widgets:
- `QLineEdit`: name
- `QComboBox`: role (`Gear`, `Hoop`, `Track`)
- `QPushButton`: Export

Export produces a `ShapeAsset` containing:
- canonical RSDL
- compiled curve/graph
- computed metrics
- role metadata

Compatibility requirement:
- exported analytic curves must implement the same interface expected by existing ring/gear curves.
- modular graphs may export as graphs initially; route export can be added later.

---

## 8. Controller, State Model, and Signals

Recommended separation:
- View: widgets and layouts
- Controller: compilation orchestration and debounce
- State: variants list, selected variant, last compile result

Signals (examples):
- compile_started
- compile_succeeded(result)
- compile_failed(diagnostics)
- variant_list_changed
- preview_settings_changed

---

## 9. Performance and Threading

- Most compiles are fast enough for the UI thread.
- Numeric-heavy operations (ellipse LUT at high resolution, huge sampling) may be moved to:
  - `QThreadPool` + `QRunnable`

Preview performance:
- Use downsampling when needed.
- Maintain separate “preview sampling density” vs “export sampling density” if required.

---

## 10. Design Guarantees

- The GUI never mutates geometry directly; it edits RSDL and parameters.
- RSDL is authoritative.
- Any previewable analytic shape is exportable.
- No changes are required to the existing rolling solver to adopt exported analytic shapes.
