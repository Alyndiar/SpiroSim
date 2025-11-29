## SpiroSim – Rolling wheel on arbitrary track

### Goal

Implement the pen trajectory for a gear wheel rolling without slipping on an arbitrary track defined by line segments and circular arcs.  
The pen position must be computed as a function of:

- the **distance traveled along the track** `s`
- the **side of the track** on which the wheel center sits
- the wheel radius and pen offset

The implementation must follow the exact geometric model below.

---

### 1. Mathematical model

Let the **contact curve** (where the teeth touch) be a 2D curve parameterized by arc length:

- \(\gamma(s) = (x_b(s), y_b(s))\): position on the track at distance \(s\) from the start  
- \(\mathbf T(s)\): **unit tangent** at \(\gamma(s)\), pointing in the direction of increasing \(s\)  
- \(\mathbf N(s)\): **unit normal**, obtained from \(\mathbf T\) by a fixed convention

**Convention for the normal (must be used consistently everywhere):**

If \(\mathbf T(s) = (T_x, T_y)\), then **define**:
\[
\mathbf N(s) = (-T_y,\, T_x)
\]

This means \(\mathbf N\) points to the **left side** of the track when moving in the direction of increasing \(s\).

Wheel and pen parameters:

- `r`: wheel radius (distance from center to contact line)
- `d`: pen offset from the wheel center
- `side`: +1 or −1, indicates on which side of the track the wheel center lies
  - `side = +1` → center on the side of \(\mathbf N(s)\)
  - `side = -1` → center on the opposite side
- `epsilon`: +1 or −1, controls the **rotation direction** of the wheel relative to the increasing `s`
- `alpha0`: initial angle of the pen ray in the local `(N, T)` frame at `s = 0`

**No-slip condition:**

When the wheel rolls **without slipping**, its rotation angle is proportional to the distance traveled:
\[
\psi(s) = \epsilon \frac{s}{r}
\]

\(\epsilon = +1\) or \(-1\) depending on the rolling direction convention.

**Wheel center position:**

The center is offset from the contact point along the normal by distance `r`:
\[
C(s) = \gamma(s) + \text{side} \cdot r \cdot \mathbf N(s)
\]

**Pen angle in local frame:**

Define the local pen angle as:
\[
\alpha(s) = \alpha_0 + \psi(s)
          = \alpha_0 + \epsilon \frac{s}{r}
\]

**Pen position (final formula):**

\[
P(s) =
  \gamma(s)
  + \text{side} \cdot r \cdot \mathbf N(s)
  + d \bigl[\cos(\alpha(s))\,\mathbf N(s)
           + \sin(\alpha(s))\,\mathbf T(s)\bigr]
\]

In components (with `T = (Tx, Ty)`, `N = (Nx, Ny)`, `gamma = (xb, yb)`):

```text
Cx = xb + side * r * Nx
Cy = yb + side * r * Ny

alpha = alpha0 + epsilon * s / r

Px = Cx + d * (cos(alpha) * Nx + sin(alpha) * Tx)
Py = Cy + d * (cos(alpha) * Ny + sin(alpha) * Ty)
```

**Important:**

- `T` and `N` must be **unit vectors** (`|T| = |N| = 1`) and orthogonal (`T·N = 0`).
- The implementation must not re-derive an absolute angle from `(x, y)` to recompute cos/sin.  
  All orientation is handled through `T`, `N` and `alpha(s)`.

---

### 2. Code structure requirements

#### 2.1. Base curve abstraction

Implement or use an abstraction like:

```python
class BaseCurve:
    def eval(self, s: float) -> tuple[float, float, tuple[float, float], tuple[float, float]]:
        """
        Evaluate the rolling track at distance s.

        Returns:
            xb, yb: base point gamma(s)
            T: (Tx, Ty) unit tangent (in direction of increasing s)
            N: (Nx, Ny) unit normal, defined as N = (-Ty, Tx)
        """
        ...
```

`BaseCurve` is responsible for:

- Handling piecewise segments (lines, circular arcs, etc.)
- Keeping a cumulative arc-length parameterization
- Returning **normalized** `T` and `N` vectors

#### 2.2. Pen position function

Implement a dedicated function that uses the math above *only*:

```python
def pen_position(
    s: float,
    base_curve: BaseCurve,
    r: float,
    d: float,
    side: int,
    alpha0: float,
    epsilon: int = 1,
) -> tuple[float, float]:
    """
    Compute the pen position for a wheel of radius r rolling without slipping
    on 'base_curve', at arc-length position s.

    Parameters:
        s       : distance traveled along the track (arc length)
        base_curve: curve object providing (xb, yb, T, N) for s
        r       : wheel radius
        d       : pen offset from wheel center
        side    : +1 => center on N(s) side, -1 => center opposite to N(s)
        alpha0  : initial local pen angle in (N, T) at s = 0
        epsilon : +1 or -1, rotation direction of the wheel vs s

    Returns:
        (Px, Py): pen position in global coordinates
    """
    ...
```

Implementation must follow the formulas under **1. Mathematical model**, nothing else.

#### 2.3. No hidden trigonometry with global angles

- Do **not** recompute a global heading angle from `(Tx, Ty)` via `atan2`.
- Always work in the local `(N, T)` basis and use `alpha = alpha0 + epsilon * s / r` directly.

---

### 3. Testing requirements

Add tests to validate the implementation against known special cases.

#### 3.1. Straight-line base → trochoid

Implement `BaseCurve` as a straight line along the x-axis:

- `gamma(s) = (s, 0)`
- `T = (1, 0)`
- `N = (0, 1)`

Choose `side = +1`, set `alpha0` so that when `d = r` you recover a standard cycloid-type shape (up to translation).

Check that:

- The wheel center moves along `(s, r)` (or equivalent offset).
- The pen trajectory looks like a cycloid/trochoid.

#### 3.2. Circular base → classical hypo/epitrochoid

Implement `BaseCurve` as a circle of radius `R`, parameterized by arc length `s`:

- `theta = s / R`
- `gamma(s) = (R * cos(theta), R * sin(theta))`
- `T` and `N` built from `theta` as per the convention.

Use different combinations of `side`, `epsilon`, `alpha0`, `d` to verify that:

- You obtain shapes consistent with hypotrochoids/epitrochoids compared to reference implementations.

#### 3.3. Sanity checks

For several values of `s`, assert numerically that:

- `|T| ≈ 1`
- `|N| ≈ 1`
- `abs(dot(T, N)) ≈ 0`

This ensures the base curve implementation is consistent with the model.

---

### 4. Integration constraints

- Do not change the existing public command-line interface unless explicitly required.
- Keep comments and docstrings up to date with the math described here.
- All new functionality must be implemented on a feature branch  
  (e.g. `feature/rolling-pen-arbitrary-track`) and covered by tests before merging into `main`.
