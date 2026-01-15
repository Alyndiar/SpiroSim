# SpiroSim

**Languages:** English | [Français](localisation/fr/README.md) | [All languages](LANGUAGES.md)

A simulator/testbed for Spirograph inspired drawings. Multiple gear layers, multiple traces per layers, "Super Spirograph"-inspired custom tracks. Configurable gear sizes, path offsets, colors. Save/export designs to JSON, PNG and SVG.

## Localisation

SpiroSim ships with a localisation system for UI strings. Use **Options → Language** to
switch languages, and see [`localisation.md`](localisation.md) for the translation file
format plus instructions on adding a new language to the program and repository.

## Installation

1. Ensure you have Python 3.10+ installed.
2. (Optional) Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
   Or with Conda/Miniconda:
   ```bash
   conda create -n spirosim python=3.10
   conda activate spirosim
   ```
3. Install the GUI dependency:
   ```bash
   python -m pip install PySide6
   ```

## Usage

Launch the main application:

```bash
python SpiroSim.py
```

## GUI overview

The main window renders the drawing and exposes the following menus and dialogs.

### File menu

- **Load settings (JSON)**: import saved layers, paths, colors, and track settings.
- **Save settings (JSON)**: export the current configuration.
- **Export as SVG**: save vector output of all visible layers.
- **Export as high-res PNG**: save a raster export at a specified resolution.
- **Quit**: exit the application.

### Layers menu

- **Manage layers and paths**: open the layer/path manager to build your design.

### Options menu

- **Background color**: set the canvas background (CSS4 name, hex, or HSL tuple).
- **Canvas size and precision**: set output width/height and points per path.
- **Language**: switch the UI between French and English.

### Regenerate menu

- **Animation**: toggle the animation controls below the canvas.
- **Show track**: toggle rendering of modular track centerlines in the preview.
- **Regenerate drawing**: recompute and refresh the drawing.

### Help menu

- **Manual**: open the localized README for the current language.
- **About**: show version and license details.

### Animation controls

When animation is enabled, controls below the canvas allow you to start/pause/reset the
preview and change the animation speed (points per second, including instant mode).

## Layers and traces (paths)

A **layer** represents one gear setup and can contain multiple **paths** (traces).
Each path is drawn from the same gear motion but with its own pen hole, phase,
color, and stroke width.

### Layer settings

When you edit a layer in the manager, you can configure:

- **Name**: label used in exports and the layer list.
- **Visible**: toggle whether the layer is rendered.
- **Layer zoom**: scales all paths in the layer together.
- **Layer translate/rotate**: offset and rotate the entire layer.
- **Number of gears (2 or 3)**: choose between a 2-gear or 3-gear system.

### Gear settings

Each layer has 2 or 3 gears. Gear 1 is stationary (the base ring or modular
track), and gear 2 (and optionally 3) are moving gears. For each gear, you can
configure:

- **Name**: label displayed in the manager.
- **Type**:
  - `ring`, `wheel`, `triangle`, `square`, `bar`, `cross`, `eye`
  - `modular` (modular track base, only allowed for Gear 1)
- **Size (wheel / inner ring)**: size for the wheel or inner ring.
- **Outer size (ring)**: outer size for ring-type gears.
- **Relation**:
  - `stationary`: only valid for Gear 1 (fixed in place).
  - `inside`: wheel rolls inside the ring (hypotrochoid).
  - `outside`: wheel rolls outside the ring (epitrochoid).
- **Modular track (notation)**: only shown for Gear 1 when type is `modular`.
  This uses the custom track notation described below.
  Use the **…** button to open the modular track editor.

### Path (trace) settings

Each path defines how the pen is placed on the moving gear:

- **Name**: label displayed in the manager and exports.
- **Hole offset**: radial hole offset on the moving gear (distance from center).
- **Phase offset (track units)**: phase shift applied to the pen position.
- **Color**: CSS4 name, `#RRGGBB` hex, or `(H, S, L)` HSL tuple.
- **Stroke width**: line width in the preview and exports.
- **Path zoom**: scales only this path (multiplicative with layer zoom).
- **Path translate/rotate**: offset and rotate the path relative to the layer.

### Track testing

If Gear 1 is a modular track with a valid notation, the manager enables
**Test path** to preview the track geometry and wheel motion.

### Layer manager actions

The layer manager lets you add, edit, reorder, enable/disable, and remove layers or paths.
It also includes bulk enable/disable for all paths in a layer.

## Custom track notation

Tracks are defined by a compact algebraic notation made of blocks, written as
`letter + number` pieces separated by operators `+`, `-`, or `*`. Whitespace is
ignored, and the entire string is case-insensitive.

## Local GitVersion metadata

To keep `spirosim/_version.py` in sync after each pull, checkout, or rebase,
install the GitVersion tool and enable the repository hooks:

```bash
dotnet tool restore --tool-manifest .config/dotnet-tools.json
python scripts/setup_git_hooks.py
```

The hooks run `python scripts/update_version.py` automatically, which calls
GitVersion and rewrites `spirosim/_version.py` with the latest `fullSemVer`
and `shortSha` values.

### Operators

- `+` / `-`: sets the turn direction (left/right) for the next piece.
- `*`: jump to the next open branch created by a `y` or `b` piece.

An initial optional `+` or `-` sets the default turn direction for the first
piece. A leading `*` jumps to the next open branch (or `*n` for branch index
`n`).

### Pieces

- `A(θ)`: arc of `θ` degrees. The sign (`+`/`-`) determines left/right turn.
- `S(L)`: straight segment of length `L`.
- `E`: endcap that terminates a branch.
- `I<n>`: intersection with `n` branches (`n > 2`).

Angles and lengths can be integer or decimal values.

### Example

```
+A(90)-S(40)+E*A(90)
```

This builds a 90° left arc, a 40-unit straight, an endcap, then continues
on the next branch with another 90° left arc.

## Shape Design Lab DSL (Analytic)

Analytic expressions define single closed pitch curves that can be used as
gears, hoops, or tracks:

- `C(N)` : circle with perimeter `N`
- `R(Ni,No)` : ring reference with inner/outer perimeters
- `P<n>(T,S/C)` : rounded polygon with `n` sides
- `D(T,O/H/L)` : drop shape
- `O(T,K)` : oblong / stadium
- `L(T,A/B)` : ellipse (numerical arc-length LUT)

These expressions can be previewed and edited in the Shape Design Lab.
