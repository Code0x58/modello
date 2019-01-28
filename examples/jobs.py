"""Modello examples based on jobs."""
from functools import lru_cache

from modello import InstanceDummy, Modello
from sympy import Eq, Function, Heaviside, Piecewise, Rational, S, Symbol, oo, solve


def generate_piecewise_income_tax(year=2019):
    """Assuming band A."""
    # 2019-2020 UK tax brackets
    year_band_brackets = {
        2018: ((-oo, 0), (11850, Rational(1, 5)), (46350, Rational(2, 5)), (150000, Rational(9, 20))),
        2019: ((-oo, 0), (1250, Rational(1, 5)), (37500, Rational(2, 5)), (150000, Rational(9, 20))),
    }
    return generate_piecewise_function(year_band_brackets[year])


def generate_piecewise_national_insurance(year=2018, band="A"):
    """Assuming band A."""
    # 2018-2019 UK band A NI
    year_band_brackets = {
        2018: {
            "A": ((-oo, 0), (702 * 12, Rational(12, 100)), (3863 * 12, Rational(2, 100))),
            "B": ((-oo, 0), (702 * 12, Rational(585, 100)), (3863 * 12, Rational(2, 100))),
            "C": ((-oo, 0),),
            "H": ((-oo, 0), (702 * 12, Rational(12, 100)), (3863 * 12, Rational(2, 100))),
            "J": ((-oo, 0), (702 * 12, Rational(2, 100))),
            "M": ((-oo, 0), (702 * 12, Rational(12, 100)), (3863 * 12, Rational(2, 100))),
            "Z": ((-oo, 0), (702 * 12, Rational(2, 100))),
        }
    }
    return generate_piecewise_function(year_band_brackets[year][band])


def generate_piecewise_employer_national_insurance(year=2018, band="A"):
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
        }
    }
    return generate_piecewise_function(year_band_brackets[year][band])


@lru_cache()
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
    # Solvers aren't implemented for the Heaviside function, so convert to piecewise and factor out
    piecewise = result.rewrite(Piecewise).replace(Heaviside(0), 0)

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

    >>> job = Job("job", salary=32000, hours=253*8)
    >>> job.hourly_income.evalf(4)
    11.37
    >>> job.employer_expense.evalf(7)
    35253.49
    """

    salary = InstanceDummy("salery", rational=True, positive=True)
    hours = InstanceDummy("hours", rational=True, positive=True)
    income = salary - IncomeTaxFunction(salary) - NationalInsuranceFunction(salary)
    hourly_income = income / hours
    employer_expense = salary + EmployerNationalInsuranceFunction(salary)
