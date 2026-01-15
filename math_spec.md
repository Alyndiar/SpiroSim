# Shape Design Lab – Mathematical Specification

## 1. Coordinate and Parameter Model

- 2D Cartesian plane.
- Curves are parameterized by **arc length** `s`.
- Every compiled curve must provide:
  - `pos(s)` → 2D position
  - `tangent(s)` → unit tangent direction
- The curve length is `T` (perimeter for closed shapes).

---

## 2. Slipless Rolling Model (Arc-Length Form)

Slipless rolling is expressed using arc-length mapping:

```text
s_gear = ± s_track + offset
```

- `+` / `-` represents rolling orientation (inside vs outside, or direction conventions).
- `offset` captures starting phase.

No collision constraints are imposed by default; only continuous slipless kinematics.

---

## 3. Circle Geometry

Given circle **size** (circumference) `N`:

```text
radius R = N / (2π)
```

Arc-length to angle mapping:

```text
θ(s) = s / R
```

Position (with a chosen frame) is derived from `cos(θ), sin(θ)`.

---

## 4. Rounded Polygon / Digon (P<n>)

### 4.1 Common Definitions

- `T` = total perimeter
- `S` = side-circle size (circumference)
- `C` = corner-circle size (circumference)
- Convert circle size to radius when needed:

```text
R_S = S / (2π)
R_C = C / (2π)
```

The curve is built from circular arcs with sweep angles chosen to satisfy:
- closure (total turning)
- perimeter constraint

All angles below are in **degrees**.

---

### 4.2 General Rounded Polygon (n ≥ 3)

Per-vertex turning budget:

```text
α + β = 360 / n
```

Perimeter constraint:

```text
T = n * (S * α/360 + C * β/360)
```

Solve for `α`:

```text
α = 360 * (T - C) / (n * (S - C))
β = 360/n - α
```

Validation:
- `0 < α < 360/n`
- `0 < β < 360/n`

Segment construction:
- Repeat `n` times alternating side arc (size `S`, sweep `α`) and corner arc (size `C`, sweep `β`) according to the chosen canonical ordering.

---

### 4.3 Digon / Football (n = 2)

Digon turning budget:

```text
α + β = 180
```

Perimeter constraint:

```text
T = 2 * (S * α/360 + C * β/360)
```

Solve:

```text
α = 180 * (T - C) / (S - C)
β = 180 - α
```

Segment order (canonical):

```text
S → C → S → C
```

This produces the football/lens shape used for SuperSpirograph-style “football gears”.

---

## 5. Drop Shape D(T, O/H/L)

Defined by a fixed sweep pattern:

```text
H:180 → L:60 → O:60 → L:60
```

Perimeter validation rule:

```text
T = H/2 + (O + 2L)/6
```

Interpretation:
- `H:180` contributes `H/2` to perimeter
- Each 60° arc contributes `size/6`

---

## 6. Oblong O(T, K)

Two semicircles plus two straights:

```text
K:180 → straight → K:180 → straight
```

Straight length per side:

```text
L_straight = (T - K) / 2
```

Validation:
- `T >= K`

---

## 7. Ellipse L(T, A/B)

Ellipse parametric form (unscaled):

```text
x = A cos(t)
y = B sin(t)
```

Arc length is not closed form in elementary functions; use numeric LUT:
- sample `t` densely
- accumulate chord lengths to approximate arc-length mapping
- scale final curve so total perimeter is `T`

Diagnostics should expose:
- LUT resolution
- perimeter residual error

---

## 8. Modular Tracks (A/S/E/I<n>)

### 8.1 Geometry

- Initial implementation uses **centerline geometry**.
- Track width derives from reference ring:

```text
width = No - Ni   (in perimeter-units mapped to radii as needed)
```

Arc pieces `A(θ)` should be interpreted as a circular arc at the appropriate reference radius (implementation-defined, but consistent).

### 8.2 Topology and Loop Closure

Loop closure and branch matching use tolerances:
- position tolerance `ε_pos`
- tangent/heading tolerance `ε_ang`

Closure error is reported as:
- distance between end connector and matched connector
- angular mismatch

---

## 9. Numerical Guarantees

- Closure error is measurable and reported.
- Tangent continuity is enforced at arc boundaries (within tolerance).
- Sampling density is configurable (Draft/Normal/High).

Self-intersections are permitted; correctness is based on continuity and arc-length parameterization.

---

## 10. Shape Preservation When Changing T

To keep silhouette “the same” while changing `T`:

1) **Uniform scaling** (preferred):
- scale all size parameters by `k = T_new / T_old`

2) **Constraint-preserving re-solve**:
- preserve sweep angles (e.g., α/β in P2)
- solve for a chosen parameter (e.g., `S`) given fixed `C` and new `T`

Both approaches must be supported as utilities (at least in the Lab UI workflow).
