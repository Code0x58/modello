"""Checks for the jobs example."""

from sympy import Rational

from examples.jobs import Job


def test_job():
    """The job model can be used to work out what salary is needed to factor out expenses."""
    expenses = 1_200
    job_a = Job("job_a", salary=26_000, hours=(253 - 25) * 9, expenses=expenses)
    # job_b adjusts the salary to make the income as though expenses were 0
    job_b = Job("job_b", income=job_a.income + job_a.expenses, hours=job_a.hours, expenses=job_a.expenses)
    salary_difference = job_b.salary - job_a.salary
    assert salary_difference > expenses
    assert salary_difference == Rational(30000, 17)  # ~1764.71
