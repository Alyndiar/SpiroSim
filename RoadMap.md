# SpiroSim Roadmap (Post-Canvas Clamp)

## 0) Current Baseline (for shared context)
- The app’s main UI is QWidget-based, with a QMenuBar, menu actions, and a central SVG viewport managed in `SpiroWindow`.
- `update_svg()` renders by calling `layers_to_svg(...)`, which builds `<path>` data and splits long paths into chunks.
- Export paths (`export_svg` / `export_png`) use `layers_to_svg(...)` with full canvas size and remain separate from onscreen rendering size.
- Animation controls live in `SpiroWindow` and there is a separate `TrackTestDialog` for track animation.
- Math backends are selected and registered in `spirosim_math.py`, and the GUI calls `generate_trochoid_points_for_layer_path(...)` regardless of backend.

---

## 1) Plan A — QWidget UI Redesign (First Phase)

### 1.1 Layout & Structural Changes
**Goal:** Right-side viewport, left-side control column, top-to-bottom flow with list → details → preview.

**Steps:**
1. Replace the top-level layout in `SpiroWindow` with a horizontal split:
   - **Left column:** menu area + layer/path list + details + preview (S=0).
   - **Right column:** viewport only (SVG widget).
2. Move the menu into the left panel:
   - Replace `QMenuBar` usage with a left-panel button/toolbar stack.
   - Keep the same actions (load/save/export, etc.).
3. Move the layer/path list from `LayerManagerDialog` into the main window:
   - Extract the list widget and embed it as a persistent left-panel element.
4. Add a “Details/Inspector” panel between list and preview:
   - Reuse the field structures from `LayerEditDialog` / `PathEditDialog`.
   - Decide whether this panel is read-only or editable.
5. Add an S=0 preview widget at the bottom-left:
   - Use math helpers to evaluate only at `s=0`.
   - Render stationary + rotating parts (no trace).

### 1.2 Dialog Behavior Improvements
**Goal:** Enter = OK, Esc = Cancel everywhere.

**Steps:**
- For each dialog, set default/auto-default on OK, and ensure Escape triggers reject:
  - `btn_ok.setDefault(True)`
  - `btn_ok.setAutoDefault(True)`

### 1.3 Remove Main Display Animation
**Goal:** Main display is static; animation only in TrackTest.

**Steps:**
- Disable/remove animation controls in `SpiroWindow`.
- Keep `TrackTestDialog` animation as-is.

### 1.4 Preserve Portability During Redesign
- Ensure the UI still calls `generate_trochoid_points_for_layer_path(...)` and only consumes points.
- Keep math/UI separation intact.

---

## 2) Plan B — QtQuick / QML Scene Graph Transition (Second Phase)

### 2.1 Why separate
- Switching to QML is a full UI stack rewrite; keep it separate from the QWidget redesign to reduce risk.

### 2.2 Migration Strategy
1. Preserve the math API contract (backend registry + point list output).
2. Rebuild the UI in QML with the same left panel + right viewport layout.
3. Replace SVG display with a SceneGraph-based renderer.
4. Keep SVG export in Python for file output.

### 2.3 Portability Considerations
- Maintain a clean separation: QML calls the same backend API and renders points.
- No additional coupling to Python-only UI details.

---

## 3) Plan C — Rust/C++ → WASM Backend (Third Phase)

### 3.1 Backend Contract (keep stable now)
- **Input:** layer/path config + points_per_path.
- **Output:** flat point list (x0, y0, x1, y1, …).

### 3.2 Steps for WASM Backend
1. Define a portable data schema (JSON/struct) equivalent to LayerConfig/PathConfig.
2. Implement a Rust/C++ version of point generation using that schema.
3. Expose as a WASM function returning a float array.
4. Add a backend adapter for web builds that calls the WASM module.
5. Keep Python/Numba backends intact for desktop.

### 3.3 Portability Guarantee
- Because the UI only requires points, the backend can be swapped without redesigning the UI.

---

## 4) Plan D — Optimization & Performance Roadmap (Remaining Points)

### 4.1 LOD + Caching
- Implement point cache per path keyed by config.
- Use lower `points_per_path` while interacting; swap to full precision at rest.

### 4.2 SVG Path Chunking (already implemented)
- Keep splitting long paths into chunks to avoid SVG truncation.

### 4.3 Faster Display Rendering (Optional Later)
- If performance is a bottleneck, move onscreen display to a GPU-friendly path:
  - QWidget: QOpenGLWidget or QGraphicsView with a custom item.
  - QML: SceneGraph renderer.
- Keep SVG export unchanged.

### 4.4 Multi-Threaded Generation
- Offload `generate_trochoid_points_for_layer_path(...)` to a worker thread and update UI on completion.

### 4.5 Further Backend Isolation
- Extract a math-only core module if deeper separation is desired for future portability.

---

## 5) Recommended Execution Order
1. QWidget redesign (layout, list, details, preview, dialog defaults, remove main animation).
2. Stabilize and validate the redesigned UI.
3. QML/QtQuick migration (rendering focus).
4. Rust/C++ WASM backend (portable math engine).
5. Performance upgrades (LOD, caching, GPU display, threading).

---

## 6) Portability & Serverless Web End-Goal
- Keep the math backend interface stable (input config → point list).
- Avoid binding UI to Python-specific constructs.
- Plan for a WASM backend plus WebGL/WebGPU rendering for the serverless web target.
