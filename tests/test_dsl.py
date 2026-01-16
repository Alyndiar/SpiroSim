from shape_dsl import ArcPiece, PolygonSpec, normalize_dsl_text, parse_analytic_expression, parse_modular_expression


def test_normalize_dsl_text_uppercases_and_strips():
    assert normalize_dsl_text(" c( 96 ) ") == "C(96)"


def test_parse_analytic_expression_case_insensitive():
    spec = parse_analytic_expression("p4(120,96/24)")
    assert isinstance(spec, PolygonSpec)
    assert spec.sides == 4
    assert spec.perimeter == 120


def test_parse_modular_expression_case_insensitive():
    spec = parse_modular_expression("+a90-l10e")
    assert spec.steps
    assert isinstance(spec.steps[0].piece, ArcPiece)
