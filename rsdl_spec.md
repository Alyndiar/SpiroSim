# Shape Design Lab – Algebraic RSDL Specification

## 1. Purpose

This document defines the **algebraic rolling surface definition language (RSDL)** used by SpiroSim’s Shape Design Lab to describe **gears, hoops, and tracks**.

The RSDL is the **single source of truth** for shape definition. All geometry is derived from it.

---

## 2. General Conventions

- All numeric values are **real numbers**.
- Units are **unitless**.
- Tooth pitch is treated as **1**.
- “Size” refers to **arc-length perimeter** (or arc-length along a segment).
- Shapes are assumed **continuous** and intended for **slipless rolling** in the solver.
- Whitespace is ignored.
- RSDL is case-insensitive; canonical form uses **uppercase** identifiers.

### 2.1 Roles (Usage Context)

The same geometric curve can be used as:
- **Gear** (rolling on the outside of another curve)
- **Hoop** (an inner boundary a gear rolls inside)
- **Track** (modular piecewise path)

The RSDL itself does not encode role; role is assigned by the caller/UI at export time.

---

## 3. Analytic Closed Shapes (Single Pitch Curve)

Analytic shapes compile to a **single closed pitch curve** (`Curve2D`).

### 3.1 Circle

```text
C N
```

- `N` : total perimeter (circumference)

---

### 3.2 Ring (Reference Shape)

```text
R(Ni, No)
```

- `Ni` : inner perimeter
- `No` : outer perimeter
- Constraint: `No > Ni`
- Primarily used as the **reference** for modular track generation (defines width and baseline curvature), but may also be previewed.

---

### 3.3 Rounded Polygon (including Digon/Football)

```text
P<n>(T, S/C)
```

- `n` : number of sides/vertices; **n ≥ 2** allowed
  - `n = 2` is a **digon** (“football/lens”)
- `T` : total perimeter
- `S` : side curvature size (perimeter of the “side circle”)
- `C` : corner curvature size (perimeter of the “corner circle”)

Notes:
- `S` and `C` are **circle sizes** (circumferences), not radii.
- `S != C` is required.
- Values may be fractional.

Examples:
```text
P2(64, 90/8)
P4(120, 96/24)
```

---

### 3.4 Drop / Squashed Drop

```text
D(T, O/H/L)
```

- `T` : total perimeter
- `O` : opposite-side curvature size
- `H` : half-circle curvature size
- `L` : link curvature size

Perimeter consistency constraint (used as validation):
```text
T = H/2 + (O + 2L)/6
```

Examples:
```text
D(72, 30/66/102)
D(84, 156/12/??)   # if extended later to allow missing parameter solving
```

---

### 3.5 Oblong / Stadium

```text
O(T, K)
```

- `T` : total perimeter
- `K` : end-cap curvature size (the two semicircles together have perimeter `K`)

Constraint:
- `T >= K`

Example:
```text
O(96, 40)
```

---

### 3.6 Ellipse

```text
L(T, A/B)
```

- `T` : target perimeter
- `A`, `B` : axis ratio values (positive)

Notes:
- Implemented numerically (arc-length LUT), then scaled to perimeter `T`.

Examples:
```text
L(120, 2/1)
L(100, 1/1)  # circle-like
```

---

## 4. Modular Track RSDL (Piecewise + Topology)

Modular track expressions compile to a **curve graph** (`CurveGraph2D`) and optionally to one or more traversable `Curve2D` routes.

### 4.1 Required Reference Ring

A modular track compile requires a reference ring to define:
- track width
- a baseline circle from which arc pieces derive their curvature

```text
R(Ni, No)
```

Constraint:
- `No > Ni`

---

### 4.2 Track Pieces

| Piece | Syntax | Parameters | Meaning |
|---|---|---|---|
| Arc | `Aθ` | `θ` degrees | Arc segment swept by `θ` degrees |
| Straight | `Lx` (or `Sx`) | `x` length | Straight segment length `x` |
| Endcap | `E` | none | Terminates a branch (adds a closing cap in track topology) |
| Intersection | `I<n>` | `n` branches | n-way junction (`n > 2`) that creates new open branches |

Notes:
- In modular mode, arc/straight are **centerline** definitions initially.
- Track width is derived from `R(Ni, No)` and applied later for ribbon rendering.

---

### 4.3 Operators and Connection Semantics

| Operator | Meaning |
|---|---|
| `+` | Connect next piece using its connector **A** |
| `-` | Connect next piece using its connector **B** (implies the piece is flipped/rotated to mate) |
| `*` | Jump to the **next open branch** (must be followed by `+` or `-` before the next piece) |
| `*n` | Jump to open branch index **n** (must be followed by `+` or `-` before the next piece) |

High-level connector rule:
- Each piece has two connectors: **A** (start) and **B** (end) in its local orientation.
- `+` means “attach next piece’s A to current open connector”.
- `-` means “attach next piece’s B to current open connector, then treat next piece’s A as the new open connector” (effectively reversing the piece’s direction).

---

### 4.4 Open Branch Model

The compiler maintains:
- `Open[]` : list of open connectors (branches)
- `b` : current branch index

Rules:
- Branch 0 starts at the first connector of the first piece.
- Without intersections, Branch 1 is the final open connector after the last piece.
- When `I<n>` is attached:
  - It consumes the current branch by connecting the intersection stem.
  - It then creates `n-1` new branches:
    - the first replaces the current branch
    - the remaining are appended to `Open[]`
- When `E` is placed, the current branch is closed/removed.
- After each piece placement, the compiler checks whether the resulting end connector:
  - closes a loop (matches the start connector), or
  - matches another open connector
  If so, those branches are removed/merged and the next open branch is selected.

---

## 5. Canonical Formatting and Round-Trip Editing

The system should support:
- RSDL text → AST → canonical RSDL text
- UI parameter edits → AST update → canonical RSDL text update

Canonical formatting rules (recommended):
- No extraneous whitespace
- `P4(120,96/24)` rather than `P4(120, 96 / 24)`
- Always use parentheses, commas and slashes exactly as specified
- Case-insensitive
- `P4(120,96/24)` rather than `p4(120,96/24)`

---

## 6. Design Guarantees

- Any valid RSDL expression produces:
  - a closed `Curve2D` (analytic shapes), or
  - a `CurveGraph2D` (modular tracks)
- The RSDL does not encode UI, rendering style, or solver settings.
- Self-intersections are permitted; correctness is based on arc-length continuity and tangent continuity.
