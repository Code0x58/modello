"""Export mathematical notation without replacing any expression identities."""

import pytest
from sympy import Eq, Integral, Symbol, latex

from modello import Index, InstanceDummy, Modello, Vector, component, inspect_model, vector


class Window(Modello):
    samples = Vector("x", size=3, real=True)
    total = samples.sum()
    i = Index("i")
    changes = vector(samples[i] - samples[i - 1], (i, 1, 2))


def test_native_sympy_latex_uses_columns_and_indexed_scalars():
    assert latex(Vector(1, 2, 3)) == r"\begin{bmatrix}1\\ 2\\ 3\end{bmatrix}"
    assert latex(Vector()) == r"\left[\right]"
    rendered = latex(Vector("x", size=3).sum())
    assert r"\sum" in rendered and r"\begin{bmatrix}" in rendered
    assert "Vector" not in rendered


def test_model_latex_preserves_vector_identity_and_formula_structure():
    problem = inspect_model(Window, "export", samples=[1, 2, 3])
    formula = problem.constraints[0].right
    assert problem.latex(problem.symbols.samples) == r"\vec{\mathrm{samples}}"
    assert problem.latex(problem.symbols.samples[1]) == r"\mathrm{samples}_{1}"
    assert problem.latex(formula) == r"\sum_{i=0}^{2} \mathrm{samples}_{i}"
    equation = Eq(problem.symbols.total, formula, evaluate=False)
    assert problem.latex(equation) == r"\mathrm{total} = \sum_{i=0}^{2} \mathrm{samples}_{i}"
    assert formula is problem.constraints[0].right


def test_whole_vector_definitions_remain_raw_scoped_and_read_only():
    class Parent(Modello):
        window = component(Window)
        alias = window.samples
        empty = vector(Window.i, (Window.i, 0, -1))

    problem = inspect_model(Parent, "formula")
    definitions = problem.vector_definitions
    assert set(definitions) == {("window", "changes"), ("empty",)}
    changes = definitions[("window", "changes")]
    samples = problem.symbols.window.samples
    assert changes[0] == samples[1] - samples[0]
    assert changes.free_symbols <= set(samples)
    assert problem.format(changes).startswith("vector(")
    rendered = problem.latex(changes)
    assert r"\middle|" in rendered and r"\in \{1, 2\}" in rendered
    assert "alias" in rendered
    assert problem.latex(samples) == r"\vec{\mathrm{alias}}"
    assert r"\varnothing" in problem.latex(definitions[("empty",)])
    with pytest.raises(TypeError):
        definitions[("empty",)] = Vector()


def test_vector_definitions_rebind_refined_components_and_export_without_evaluation(monkeypatch):
    from sympy import Sum

    class Positive(Window):
        samples = Vector("x", size=3, positive=True)

    class Parent(Modello):
        window = component(Window)

    class Refined(Parent):
        window = component(Positive)

    def forbidden(*args, **kwargs):
        raise AssertionError("Formula export must not evaluate operators")

    monkeypatch.setattr(Sum, "doit", forbidden)
    problem = inspect_model(Refined, "raw")
    changes = problem.vector_definitions[("window", "changes")]
    samples = problem.symbols.window.samples
    assert changes.free_symbols == set(samples)
    assert all(symbol.is_positive for symbol in samples)
    assert "window.samples" in problem.latex(changes)
    assert r"\sum" in problem.latex(problem.constraints[0].right)


def test_comprehension_export_preserves_offset_domains_and_inlines_temporary_selections():
    i = Index("i")
    squares = vector(i**2, (i, 2, 5))
    assert latex(squares) == r"\left[i^{2}\;\middle|\;i\in \{2, \ldots, 5\}\right]"
    assert latex(squares.sum()) == r"\sum_{j=0}^{3} \left(j + 2\right)^{2}"
    large = vector(i**2, (i, 0, 1_000_000))
    assert latex(large) == r"\left[i^{2}\;\middle|\;i\in \{0, \ldots, 1000000\}\right]"
    samples = Vector("x", size=3)
    rendered = latex((2 * samples).sum())
    assert "Vector" not in rendered
    assert r"\sum" in rendered


def test_latex_binder_labels_do_not_change_an_external_parameter_with_the_same_identity():
    t = Symbol("t", real=True)

    class Area(Modello):
        height = InstanceDummy("height", real=True)
        area = Integral(t, (t, 0, height))

    problem = inspect_model(Area, "bound")
    expression = t + problem.constraints[0].right
    rendered = problem.latex(expression)
    assert r"\mathrm{external[t]}" in rendered
    assert r"\int\limits_{0}^{\mathrm{height}} t\, dt" in rendered


def test_exported_path_labels_escape_tex_metacharacters():
    class Named(Modello):
        average_cost = InstanceDummy("cost", real=True)
        quantity = InstanceDummy("quantity", real=True)

    problem = inspect_model(Named, "literal")
    assert problem.latex(problem.symbols.average_cost) == r"\mathrm{average\_cost}"
    assert problem.latex(problem.symbols.average_cost * problem.symbols.quantity) == (
        r"\mathrm{average\_cost} \cdot \mathrm{quantity}"
    )
    external = Symbol(r"rate_%{x}\input{file}")
    rendered = problem.latex(external)
    assert r"\input{" not in rendered
    assert r"\%\{" in rendered and r"\backslash " in rendered


def test_family_latex_accepts_both_public_symbols_and_internal_chart_coordinates():
    family = inspect_model(Window, "family").solutions()
    assert family.latex(family.symbols.samples) == r"\vec{\mathrm{samples}}"
    chart = Vector(*(binding.expression for binding in family._bindings if binding.field.path[0] == "samples"))
    assert family.latex(chart) == r"\vec{\mathrm{samples}}"
    assert family.format(family.symbols.samples) == family.format(chart) == "samples"
    assert family.format(family.problem.constraints[0].right) == "Sum(samples[i], (i, 0, 2))"
    observed = family.observe(samples=[1, 2, 3])
    assert observed.latex(observed.expression(observed.symbols.samples)) == r"\begin{bmatrix}1\\ 2\\ 3\end{bmatrix}"


def test_latex_names_customize_vector_coordinates_with_explicit_scalar_precedence():
    problem = inspect_model(Window, "named")
    samples = problem.symbols.samples
    names = {samples: "x"}
    assert problem.latex(samples, vector_names=names) == r"\vec{x}"
    assert problem.latex(samples[1], vector_names=names) == "x_{1}"
    assert problem.latex(samples[1], vector_names=names, symbol_names={samples[1]: r"\xi"}) == r"\xi"
    formula = Eq(problem.symbols.total, problem.constraints[0].right, evaluate=False)
    assert problem.latex(formula, vector_names=names, symbol_names={problem.symbols.total: "S"}) == (
        r"S = \sum_{i=0}^{2} x_{i}"
    )
    family = problem.solutions()
    chart = Vector(*(binding.expression for binding in family._bindings if binding.field.path[0] == "samples"))
    assert family.latex(chart, vector_names=names) == r"\vec{x}"


def test_a_repeated_vector_coordinate_retains_one_canonical_export_name():
    class Repeated(Modello):
        x = InstanceDummy("x", real=True)
        repeated = Vector(x, x)

    problem = inspect_model(Repeated, "alias")
    values = problem.symbols.repeated
    assert problem.latex(values[0], vector_names={values: "v"}) == "v_{0}"
    assert problem.latex(values[1], vector_names={values: "v"}) == "v_{0}"


def test_vector_aliases_prefer_the_current_component_in_both_printing_contexts():
    class Parent(Modello):
        window = component(Window)
        alias = window.samples

    problem = inspect_model(Parent, "scope")
    formula = problem.constraints[0].right
    family = problem.solutions()
    assert problem.format(formula) == family.format(formula) == "Sum(alias[i], (i, 0, 2))"
    expected = "Sum(samples[i], (i, 0, 2))"
    assert problem._printer(formula, scope=("window",)).doprint(formula) == expected
    assert family._printer(formula, scope=("window",)).doprint(formula) == expected
