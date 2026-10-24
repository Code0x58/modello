"""A scalar recurrence with an independently observed finite window."""

from modello import Index, InstanceDummy, Modello, Recurrence, Series, Vector


class ObservedAutoregression(Modello):
    """A relative law with an independently chosen observation window."""

    N = InstanceDummy("N", integer=True, nonnegative=True)
    opening = InstanceDummy("opening", real=True)
    retention = InstanceDummy("retention", real=True)
    innovations = Vector("epsilon", size=N, real=True)
    n = Index("n", nonnegative=True)
    T = Series("T", n, real=True)
    law = Recurrence(T[n], retention * T[n - 1] + innovations[n - 1], initial={0: opening})
    levels = law.window(0, N)
    closing = T[N]
