"""Modello examples based on jobs."""
from functools import lru_cache

from modello import InstanceDummy, Modello
from sympy import Eq, Function, Heaviside, Piecewise, Rational, S, Symbol, oo, solve

YEAR = 2023


def generate_piecewise_income_tax(year=YEAR):
    """Assuming band A."""
    # 2019-2020 UK tax brackets
    # XXX: this omits the income limit for personal allowance. A fix for this could be to add an additional
    #  bracket from £100,000 to £125,000 (for 2019)
    year_band_brackets = {
        2018: (
            (-oo, 0),
            (11850, Rational(1, 5)),
            (46350, Rational(2, 5)),
            (150000, Rational(9, 20)),
        ),
        2019: (
            (-oo, 0),
            (12500, Rational(1, 5)),
            (37500, Rational(2, 5)),
            (150000, Rational(9, 20)),
        ),
        2022: (
            (-oo, Rational(1, 5)),
            (37_700, Rational(2, 5)),
            (150_000, Rational(9, 20)),
        ),
        2023: (
            (-oo, 0),
            (12_570, Rational(1, 5)),
            (50_270, Rational(2, 5)),
            (125_140, Rational(9, 20)),
        ),
    }
    return generate_piecewise_function(year_band_brackets[year])


def generate_piecewise_national_insurance(year=YEAR, band="A"):
    """Assuming band A."""
    # 2018-2019 UK band A NI
    year_band_brackets = {
        2018: {
            "A": (
                (-oo, 0),
                (702 * 12, Rational(12, 100)),
                (3863 * 12, Rational(2, 100)),
            ),
            "B": (
                (-oo, 0),
                (702 * 12, Rational(585, 100)),
                (3863 * 12, Rational(2, 100)),
            ),
            "C": ((-oo, 0),),
            "H": (
                (-oo, 0),
                (702 * 12, Rational(12, 100)),
                (3863 * 12, Rational(2, 100)),
            ),
            "J": ((-oo, 0), (702 * 12, Rational(2, 100))),
            "M": (
                (-oo, 0),
                (702 * 12, Rational(12, 100)),
                (3863 * 12, Rational(2, 100)),
            ),
            "Z": ((-oo, 0), (702 * 12, Rational(2, 100))),
        },
        2019: {
            "A": (
                (-oo, 0),
                (719 * 12, Rational(12, 100)),
                (4167 * 12, Rational(2, 100)),
            ),
            "B": (
                (-oo, 0),
                (219 * 12, Rational(585, 100)),
                (4167 * 12, Rational(2, 100)),
            ),
            "C": ((-oo, 0),),
            "H": (
                (-oo, 0),
                (719 * 12, Rational(12, 100)),
                (4167 * 12, Rational(2, 100)),
            ),
            "J": ((-oo, 0), (719 * 12, Rational(2, 100))),
            "M": (
                (-oo, 0),
                (719 * 12, Rational(12, 100)),
                (4167 * 12, Rational(2, 100)),
            ),
            "Z": ((-oo, 0), (719 * 12, Rational(2, 100))),
        },
        2023: {
            "A": (
                (-oo, 0),
                (1048 * 12, Rational(12, 100)),
                (4189 * 12, Rational(2, 100)),
            ),
            "B": (
                (-oo, 0),
                (1048 * 12, Rational(585, 100)),
                (4189 * 12, Rational(2, 100)),
            ),
            "C": ((-oo, 0),),
            "H": (
                (-oo, 0),
                (1048 * 12, Rational(12, 100)),
                (4189 * 12, Rational(2, 100)),
            ),
            "J": ((-oo, 0), (1048 * 12, Rational(2, 100))),
            "M": (
                (-oo, 0),
                (1048 * 12, Rational(12, 100)),
                (4189 * 12, Rational(2, 100)),
            ),
            "Z": ((-oo, 0), (1048 * 12, Rational(2, 100))),
        },
    }
    return generate_piecewise_function(year_band_brackets[year][band])


def generate_piecewise_employer_national_insurance(year=YEAR, band="A"):
    """Assuming band A."""
    # 2018-2019 UK band A NI
    year_band_brackets = {
        2018: {
            "A": ((-oo, 0), (702 * 12, Rational(138, 1000))),
            "B": ((-oo, 0), (702 * 12, Rational(138, 1000))),
            "C": ((-oo, 0), (702 * 12, Rational(138, 1000))),
            "H": ((-oo, 0), (3863 * 12, Rational(138, 1000))),
            "J": ((-oo, 0), (702 * 12, Rational(138, 1000))),
            "M": ((-oo, 0), (3863 * 12, Rational(138, 1000))),
            "Z": ((-oo, 0), (3863 * 12, Rational(138, 1000))),
        },
        2019: {
            "A": ((-oo, 0), (719 * 12, Rational(138, 1000))),
            "B": ((-oo, 0), (719 * 12, Rational(138, 1000))),
            "C": ((-oo, 0), (719 * 12, Rational(138, 1000))),
            "H": ((-oo, 0), (4167 * 12, Rational(138, 1000))),
            "J": ((-oo, 0), (719 * 12, Rational(138, 1000))),
            "M": ((-oo, 0), (4167 * 12, Rational(138, 1000))),
            "Z": ((-oo, 0), (4167 * 12, Rational(138, 1000))),
        },
        2023: {
            "A": ((-oo, 0), (1048 * 12, Rational(138, 1000))),
            "B": ((-oo, 0), (1048 * 12, Rational(138, 1000))),
            "C": ((-oo, 0), (1048 * 12, Rational(138, 1000))),
            "H": ((-oo, 0), (4189 * 12, Rational(138, 1000))),
            "J": ((-oo, 0), (1048 * 12, Rational(138, 1000))),
            "M": ((-oo, 0), (4189 * 12, Rational(138, 1000))),
            "Z": ((-oo, 0), (4189 * 12, Rational(138, 1000))),
        },
    }
    return generate_piecewise_function(year_band_brackets[year][band])


@lru_cache(maxsize=None)
def generate_piecewise_function(brackets):
    """Generate a piecewise for taxes.

    >>> f = generate_piecewise_function(((-oo, 0), (10, Rational(1, 10)), (20, Rational(1, 5))))
    >>> f(10)
    0
    >>> f(20)
    1
    >>> f(30)
    3
    >>> g = generate_piecewise_function(((-oo, 0),))
    >>> g(1234)
    0
    >>> x = Symbol("x")
    >>> solve(Eq(f(x), 1))
    [20]
    """
    gross = Symbol("gross_income", rational=True, positive=True)
    # Personal Allowance, Basic Rate, Higher Rate, Additional Rate
    result = S.Zero
    for i, (limit, tax) in enumerate(brackets[1:]):
        # add the increase in tax for each limit
        result += Heaviside(gross - limit, 0) * (tax - brackets[i][1]) * (gross - limit)
    # Solvers aren't implemented for the Heaviside function, so convert to piecewise
    # FIXME: apply appropriate constraints. If gross_income is positive the expression is positive
    piecewise = result.rewrite(Piecewise)

    class F(Function):
        @classmethod
        def eval(cls, gross_income):
            return piecewise.replace(gross, gross_income)

    return F


IncomeTaxFunction = generate_piecewise_income_tax()
NationalInsuranceFunction = generate_piecewise_national_insurance()
EmployerNationalInsuranceFunction = generate_piecewise_employer_national_insurance()


class Job(Modello):
    """
    Simple model of a job in the UK.

    This assumes all the values are yearly.

    >>> job = Job("job", salary=32000, hours=253*8, expenses=0)
    >>> job.hourly_income.evalf(4)
    12.74
    >>> job.employer_expense.evalf(7)
    34680.51
    """

    salary = InstanceDummy("salery", rational=True, positive=True)
    hours = InstanceDummy("hours", rational=True, positive=True)
    expenses = InstanceDummy("expenses")
    income = (
            salary
            - IncomeTaxFunction(salary)
            - NationalInsuranceFunction(salary)
            - expenses
    )
    hourly_income = income / hours
    employer_expense = salary + EmployerNationalInsuranceFunction(salary)


def test_job():
    """The job model can be used to work out what salary is needed to factor out expenses."""
    expenses = 1_200
    job_a = Job('job_a', salary=26_000, hours=(253 - 25) * 9, expenses=expenses)
    # job_b adjusts the salary to make the income as though expenses were 0
    job_b = Job('job_b',
                income=job_a.income + job_a.expenses,
                hours=job_a.hours,
                expenses=job_a.expenses)
    salary_difference = job_b.salary - job_a.salary
    assert salary_difference > expenses
    assert salary_difference == Rational(30000, 17)  # ~1764.71
