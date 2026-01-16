import math

from shape_geometry import CircleCurve, StraightLineCurve, pen_position, roll_pen_position, wheel_pen_local_vector


def _reference_epitrochoid(theta: float, R: float, r: float, d: float) -> tuple[float, float]:
    x = (R + r) * math.cos(theta) - d * math.cos((R + r) / r * theta)
    y = (R + r) * math.sin(theta) - d * math.sin((R + r) / r * theta)
    return x, y


def _reference_hypotrochoid(theta: float, R: float, r: float, d: float) -> tuple[float, float]:
    x = (R - r) * math.cos(theta) + d * math.cos((R - r) / r * theta)
    y = (R - r) * math.sin(theta) - d * math.sin((R - r) / r * theta)
    return x, y


def test_straight_line_base_center_and_tangent():
    base = StraightLineCurve(10.0)
    r = 2.0
    d = 0.0
    side = 1
    epsilon = -1

    for s in [0.0, 1.5, 4.0, 9.5]:
        xb, yb, (tx, ty), (nx, ny) = base.eval(s)
        assert math.isclose(xb, s, abs_tol=1e-9)
        assert math.isclose(yb, 0.0, abs_tol=1e-9)
        assert math.isclose(tx, 1.0, abs_tol=1e-9)
        assert math.isclose(ty, 0.0, abs_tol=1e-9)
        assert math.isclose(nx, 0.0, abs_tol=1e-9)
        assert math.isclose(ny, 1.0, abs_tol=1e-9)

        px, py = pen_position(s, base, r, d, side, alpha0=0.0, epsilon=epsilon)
        assert math.isclose(px, s, abs_tol=1e-6)
        assert math.isclose(py, r, abs_tol=1e-6)


def test_circle_base_epitrochoid_matches_reference():
    R = 10.0
    r = 3.0
    d = 4.0
    base = CircleCurve(2.0 * math.pi * R)
    side = -1
    epsilon = -1

    for theta in [0.0, 0.5, 1.0, 1.5]:
        s = R * theta
        px, py = pen_position(s, base, r, d, side, alpha0=0.0, epsilon=epsilon)
        rx, ry = _reference_epitrochoid(theta, R, r, d)
        assert math.isclose(px, rx, abs_tol=1e-6)
        assert math.isclose(py, ry, abs_tol=1e-6)


def test_circle_base_hypotrochoid_matches_reference():
    R = 10.0
    r = 3.0
    d = 4.0
    base = CircleCurve(2.0 * math.pi * R)
    side = 1
    epsilon = 1
    alpha0 = math.pi

    for theta in [0.0, 0.5, 1.0, 1.5]:
        s = R * theta
        px, py = pen_position(s, base, r, d, side, alpha0=alpha0, epsilon=epsilon)
        rx, ry = _reference_hypotrochoid(theta, R, r, d)
        assert math.isclose(px, rx, abs_tol=1e-6)
        assert math.isclose(py, ry, abs_tol=1e-6)


def test_circle_base_tangent_normal_unit():
    base = CircleCurve(2.0 * math.pi * 5.0)
    for s in [0.0, 2.5, 7.5, 15.0]:
        _, _, (tx, ty), (nx, ny) = base.eval(s)
        assert math.isclose(math.hypot(tx, ty), 1.0, abs_tol=1e-9)
        assert math.isclose(math.hypot(nx, ny), 1.0, abs_tol=1e-9)
        assert math.isclose(tx * nx + ty * ny, 0.0, abs_tol=1e-9)


def test_roll_pen_position_matches_circle_pen_position():
    r = 2.5
    d = 1.25
    base = StraightLineCurve(12.0)
    wheel = CircleCurve(2.0 * math.pi * r)
    side = 1
    epsilon = 1
    alpha0 = 0.35
    pen_local = wheel_pen_local_vector(base, wheel, d, side, alpha0)

    for s in [0.0, 1.0, 4.5, 9.0, 11.5]:
        px, py = pen_position(s, base, r, d, side, alpha0=alpha0, epsilon=epsilon)
        rx, ry = roll_pen_position(s, base, wheel, d, side, alpha0, epsilon=epsilon, pen_local=pen_local)
        assert math.isclose(px, rx, abs_tol=1e-6)
        assert math.isclose(py, ry, abs_tol=1e-6)
