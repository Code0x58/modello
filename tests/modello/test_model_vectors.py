"""Typed ordered components share the flat solver and canonical state objects."""

import pytest
from sympy import Rational, Sum, Symbol

from modello import (
    Index,
    InstanceDummy,
    Modello,
    ModelSequence,
    ModelVector,
    Recurrence,
    Series,
    Vector,
    component,
    inspect_model,
    model_recurrence,
    model_vector,
)
from modello.errors import ModelDefinitionError, ModelInputError, SolutionCountError


class State(Modello):
    balance = InstanceDummy("balance", real=True)


class Payment(Modello):
    amount = InstanceDummy("amount", real=True)
    fee = InstanceDummy("fee", nonnegative=True)
    debit = amount + fee


class Posting(Modello):
    before = component(State)
    payment = component(Payment)
    after = component(State)
    balance = after.balance
    balance = before.balance - payment.debit


class Path(Modello):
    N = InstanceDummy("N", integer=True, nonnegative=True)
    opening = component(State)
    payments = model_vector(Payment, size=N)
    states = model_recurrence(Posting, initial=opening, payment=payments)
    terminal = states[-1]
    amounts = payments.project(lambda payment: payment.amount)
    balances = states.project(lambda state: state.balance)
    total_debit = payments.project(lambda payment: payment.debit).sum()


def observations():
    return [Payment("first", amount=10, fee=1), Payment("second", amount=20, fee=2)]


def test_model_vectors_infer_length_preserve_types_and_share_state_views():
    case = Path("path", opening=State("opening", balance=100), payments=observations())
    assert isinstance(case.payments, ModelVector)
    assert isinstance(case.states, ModelSequence)
    assert isinstance(case.payments[0], Payment)
    assert isinstance(case.states[-1], State)
    assert len(case.payments) == case.N == 2
    assert case.states.shape == (3,)
    assert case.balances == Vector(100, 89, 67)
    assert case.total_debit == 33
    assert case.states[0] is case.opening
    assert case.states[-1] is case.terminal
    assert case.states.steps[1].before is case.states[1]
    assert case.states.steps[0].payment is case.payments[0]
    assert case.payments[::-1] == (case.payments[1], case.payments[0])
    assert tuple(case.states) == (case.opening, case.states[1], case.terminal)
    assert case.payments.project(lambda payment: 2 * payment.amount).doit() == Vector(20, 40)
    i = Index("i")
    assert case.payments[i].amount.subs(i, 1) == 20


def test_terminal_observations_solve_missing_inputs_in_constructor_and_families():
    values = {
        "opening": State("opening", balance=100),
        "payments": [Payment("first", amount=10, fee=1), Payment("second", fee=2)],
        "terminal": State("target", balance=67),
    }
    case = Path("inverse", **values)
    assert case.payments[1].amount == 20
    problem = inspect_model(Path, "family", **values)
    assert problem.field("terminal", "balance").symbol == problem.field("states", 2, "balance").symbol
    assert problem.symbols.payments[1].amount == problem.field("payments", 1, "amount").symbol
    assert problem.format(problem.symbols.payments[1].amount) == "amounts[1]"
    selected = problem.solutions().one()
    assert selected.payments[1].amount == 20
    assert selected.terminal is selected.states[-1]
    assert not any(c.path[0] == "terminal" and c.kind == "definition" for c in problem.constraints)


def test_empty_path_and_unknown_members_keep_correct_boundaries_and_scopes():
    empty = Path("empty", opening=State("opening", balance=100), payments=[])
    assert empty.N == 0
    assert empty.payments.shape == (0,)
    assert empty.balances == Vector(100)
    assert empty.total_debit == 0
    assert empty.terminal is empty.opening
    left = Path("same", N=2)
    right = Path("same", payments=[None, None])
    assert left.payments[0].amount != left.payments[1].amount
    assert left.payments[0].amount != right.payments[0].amount
    assert left.terminal.balance == left.opening.balance - left.total_debit


def test_typed_inputs_retain_provenance_and_whole_observation_group():
    problem = inspect_model(Path, "copied", payments=observations())
    copied = [c for c in problem.constraints if c.input == "payments"]
    assert {c.kind for c in copied if c.path[-1] == "debit"} == {"inferred"}
    assert {c.kind for c in copied if c.path[-1] in {"amount", "fee"}} == {"input"}
    assert not any(c.input == "payments" for c in problem.without("payments").constraints)
    changed = observations()
    changed[0].debit = 12
    with pytest.raises(SolutionCountError):
        Path("overwritten", payments=changed)


def test_projection_uses_typed_nested_fields_and_vector_member_schema():
    class Sensor(Modello):
        samples = Vector("x", size=2, real=True)
        state = component(State)

    class Panel(Modello):
        sensors = model_vector(Sensor, size=2)
        seconds = sensors.project(lambda sensor: sensor.samples[1])
        balances = sensors.project(lambda sensor: sensor.state.balance)

    case = Panel("panel", sensors=[Sensor("a", samples=[1, 2], state=State("s", balance=3)), None])
    assert case.seconds[0] == 2
    assert case.balances[0] == 3
    assert case.seconds[1] == case.sensors[1].samples[1]
    assert case.balances[1] == case.sensors[1].state.balance


def test_multiple_indexed_inputs_have_joint_shape_equations():
    class TwoPayments(Posting):
        second = component(Payment)
        balance = Posting.before.balance - Posting.payment.debit - second.debit

    class PairPath(Modello):
        N = InstanceDummy("N", integer=True, nonnegative=True)
        M = InstanceDummy("M", integer=True, nonnegative=True)
        opening = component(State)
        left = model_vector(Payment, size=N)
        right = model_vector(Payment, size=M)
        states = model_recurrence(TwoPayments, initial=opening, payment=left, second=right)

    case = PairPath("shape", left=observations())
    assert len(case.right) == case.M == case.N == 2
    with pytest.raises(ModelInputError, match="conflict"):
        PairPath("bad", left=observations(), right=[None])


@pytest.mark.parametrize("values", [{}, {"N": -1}, {"N": Rational(1, 2)}, {"N": 1, "payments": observations()}])
def test_invalid_or_unresolved_structural_sizes_are_owned_errors(values):
    with pytest.raises(ModelInputError):
        Path("invalid", **values)


def test_collection_input_and_index_errors_are_owned():
    with pytest.raises(ModelInputError, match="Payment"):
        Path("wrong_member", payments=[State("state")])
    case = Path("one", N=1)
    for index in (True, Rational(1, 2), Symbol("x", integer=False)):
        with pytest.raises(ModelInputError):
            case.payments[index]
    with pytest.raises(IndexError):
        case.payments[1]
    with pytest.raises(ModelInputError):
        model_vector(Payment, size=True)


def test_recurrence_rejects_incompatible_ports_and_competing_wiring():
    class OtherState(State):
        pass

    class Wrong(Posting):
        after = component(OtherState)

    with pytest.raises(ModelInputError, match="matching"):

        class Invalid(Modello):
            opening = component(State)
            states = model_recurrence(Wrong, initial=opening, size=2)

    with pytest.raises(ModelInputError, match="distinct"):

        class Competing(Modello):
            opening = component(State)
            states = model_recurrence(Posting, initial=opening, size=2, before=opening)


def test_state_domains_and_function_identities_survive_component_wiring():
    class LawState(Modello):
        value = InstanceDummy("value", positive=True)
        n = Index("n")
        T = Series("T", n, positive=True)
        law = Recurrence(T[n], 2 * T[n - 1], initial={0: value})

    class Evolution(Modello):
        before = component(LawState)
        after = component(LawState)
        value = after.value
        value = before.law.at(1)

    class EvolutionPath(Modello):
        opening = component(LawState)
        states = model_recurrence(Evolution, initial=opening, size=2)
        terminal = states[-1]

    case = EvolutionPath("law", opening=LawState("opening", value=3))
    assert case.terminal.value == 12
    assert case.states.steps[1].before.law == case.states[1].law
    problem = inspect_model(EvolutionPath, "scope")
    assert problem.symbols.states.steps[1].before.T == problem.symbols.states[1].T
    assert problem.symbols.states[0].T != problem.symbols.states[1].T
    target = LawState("bad")
    target.value = -1
    with pytest.raises(SolutionCountError):
        EvolutionPath("negative", terminal=target)


@pytest.mark.parametrize(
    "name,value",
    [
        ("terminal", InstanceDummy("last")),
        ("terminal", State),
        ("payments", State),
        ("payments", Vector("x", size=1)),
    ],
)
def test_inheritance_rejects_collection_and_view_kind_changes(name, value):
    namespace = type(Path).__prepare__("Invalid", (Path,))
    with pytest.raises(ModelDefinitionError, match="another field kind"):
        namespace[name] = value


def test_scalar_to_collection_kind_change_is_rejected():
    class Original(Modello):
        items = InstanceDummy("items")

    with pytest.raises(ModelDefinitionError, match="model collection"):

        class Invalid(Original):
            items = model_vector(Payment, size=2)


def test_named_integer_selections_resolve_from_observations_and_definitions():
    class Pick(Modello):
        N = InstanceDummy("N", integer=True, positive=True)
        index = InstanceDummy("index", integer=True)
        last_index = N - 1
        payments = model_vector(Payment, size=N)
        chosen = payments[index]
        last = payments[last_index]
        chosen_debit = chosen.debit

    case = Pick("pick", payments=observations(), index=0)
    assert case.chosen is case.payments[0]
    assert case.last is case.payments[-1]
    assert case.chosen_debit == 11
    assert case.last_index == 1
    with pytest.raises(SolutionCountError):
        type(case)("frozen selection", payments=observations(), index=1)
    assert Pick.payments.shape == (Pick.N,)
    assert case.payments.shape == (2,)
    with pytest.raises(ModelInputError, match="integer index"):
        Pick("missing", payments=observations())
    with pytest.raises(IndexError):
        Pick("outside", payments=observations(), index=2)


def test_scalar_relative_selection_and_native_sum_resolve_symbolic_bounds():
    class Query(Path):
        closing_balance = Path.states[-1].balance
        n = Index("n")
        debit = Sum(Path.payments[n].debit, (n, 0, Path.N - 1))

    case = Query("query", payments=observations(), opening=State("opening", balance=100))
    assert case.closing_balance == case.terminal.balance == 67
    assert case.debit == case.total_debit == 33
    empty = Query("empty", payments=[], opening=State("opening", balance=100))
    assert empty.closing_balance == 100
    assert empty.debit == 0


def test_refining_dimension_domains_preserves_projected_column_identities():
    class Refined(Path):
        N = InstanceDummy("N", integer=True, positive=True)
        twice = Path.payments.project(lambda payment: 2 * payment.amount)

    case = Refined("refined", payments=observations(), opening=State("opening", balance=100))
    assert case.amounts == Vector(10, 20)
    assert case.twice == Vector(20, 40)
    assert case.balances == Vector(100, 89, 67)
    assert case.terminal.balance == 67


def test_nested_component_selections_copy_a_concrete_schema():
    class Pick(Modello):
        index = InstanceDummy("index", integer=True)
        payments = model_vector(Payment, size=2)
        chosen = payments[index]

    class Parent(Modello):
        child = component(Pick)

    with pytest.raises(ModelInputError, match="concrete"):

        class Invalid(Modello):
            members = model_vector(Pick, size=2)

    selected = Pick("selected", index=1, payments=observations())
    copied = Parent("parent", child=selected)
    assert isinstance(copied.child.chosen, Payment)
    assert copied.child.chosen is copied.child.payments[1]
    assert copied.child.chosen.debit == 22


def test_state_observations_determine_the_horizon_and_unknown_opening():
    case = Path("states", states=[State("first", balance=100), State("next", balance=89)])
    assert case.N == 1
    assert case.opening.balance == 100
    assert case.payments[0].debit == 11
    with pytest.raises(ModelInputError):
        Path("no_boundary", states=[])


def test_shared_component_inputs_and_custom_port_names():
    class Factor(Modello):
        value = InstanceDummy("factor", positive=True)

    class Growth(Modello):
        source = component(State)
        target = component(State)
        factor = component(Factor)
        balance = target.balance
        balance = source.balance * factor.value

    class GrowthPath(Modello):
        opening = component(State)
        factor = component(Factor)
        states = model_recurrence(Growth, initial=opening, size=2, previous="source", result="target", factor=factor)
        terminal = states[-1]

    result = GrowthPath("inverse", opening=State("initial", balance=2), terminal=State("target", balance=18))
    assert result.factor.value == 3
    assert result.states.steps[0].factor is result.states.steps[1].factor is result.factor


def test_nonlinear_transition_domains_select_the_positive_inverse_path():
    class PositiveState(Modello):
        value = InstanceDummy("value", positive=True)

    class Square(Modello):
        before = component(PositiveState)
        after = component(PositiveState)
        value = after.value
        value = before.value**2

    class SquarePath(Modello):
        opening = component(PositiveState)
        states = model_recurrence(Square, initial=opening, size=2)
        terminal = states[-1]

    case = SquarePath("inverse", terminal=PositiveState("target", value=81))
    assert case.opening.value == 3
    assert case.states[1].value == 9


def test_unresolved_series_values_are_rebound_when_copying_model_members():
    class ProcessState(Modello):
        process = Series("X", real=True)
        value = process[0]

    class Components(Modello):
        left = component(ProcessState)
        right = component(ProcessState)

    class VectorOfStates(Modello):
        members = model_vector(ProcessState, size=2)

    source = ProcessState("source")
    pair = Components("pair", left=source, right=source)
    assert pair.left.value == pair.left.process[0]
    assert pair.right.value == pair.right.process[0]
    assert pair.left.value != pair.right.value
    collection = VectorOfStates("vector", members=[source, source])
    assert collection.members[0].value == collection.members[0].process[0]
    assert collection.members[1].value == collection.members[1].process[0]
    assert collection.members[0].value != collection.members[1].value


def test_projection_can_select_a_fixed_member_of_a_nested_model_vector():
    class Row(Modello):
        payments = model_vector(Payment, size=2)

    class Table(Modello):
        rows = model_vector(Row, size=2)
        seconds = rows.project(lambda row: row.payments[-1].amount)

    case = Table("table", rows=[Row("first", payments=observations()), None])
    assert case.seconds[0] == 20
    assert case.seconds[1] == case.rows[1].payments[1].amount
    with pytest.raises(ModelInputError, match="concrete integer"):
        Table.rows.project(lambda row: row.payments[Index("j")].amount)


def test_chained_recurrences_resolve_the_prior_terminal_after_binding_its_horizon():
    class Chained(Modello):
        N = InstanceDummy("N", integer=True, nonnegative=True)
        opening = component(State)
        payments = model_vector(Payment, size=N)
        first = model_recurrence(Posting, initial=opening, payment=payments)
        last_payment = component(Payment)
        next = model_recurrence(Posting, initial=first[-1], size=1, payment=last_payment)
        terminal = next[-1]

    case = Chained(
        "chained",
        opening=State("opening", balance=100),
        payments=observations(),
        last_payment=Payment("extra", amount=5, fee=1),
    )
    assert case.terminal.balance == 61
    assert case.next[0] is case.first[-1]
    assert case.next.steps[0].payment is case.last_payment


def test_recurrence_sources_can_use_a_supplied_integer_component_selection():
    class SelectedInitial(Modello):
        index = InstanceDummy("index", integer=True)
        options = model_vector(State, size=2)
        payment = component(Payment)
        states = model_recurrence(Posting, initial=options[index], size=1, payment=payment)
        terminal = states[-1]

    case = SelectedInitial(
        "selected",
        index=1,
        options=[State("a", balance=200), State("b", balance=100)],
        payment=Payment("payment", amount=10, fee=1),
    )
    assert case.terminal.balance == 89
    assert case.states[0] is case.options[1]
    with pytest.raises(ModelInputError, match="integer index"):
        SelectedInitial("missing")


def test_unresolved_overridden_template_observations_require_their_concrete_schema():
    class Process(Modello):
        process = Series("X", real=True)
        value = process[0]

    class Refined(Process):
        process = Series("Y", positive=True)

    class Parent(Modello):
        child = component(Process)

    with pytest.raises(ModelInputError, match="overridden symbolic template"):
        Parent("copy", child=Refined("source"))
