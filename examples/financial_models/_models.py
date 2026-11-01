"""Built-in model choices own their study and figure behavior."""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, assert_never

if TYPE_CHECKING:
    from ._plots import _Plot
    from ._study import Study


class _BuiltinModel(StrEnum):
    BLACK_SCHOLES = "black-scholes"
    PARTICIPATION_EXECUTION = "participation-execution"
    MARKOWITZ = "markowitz"
    CIR_BOND = "cir-bond"
    PERPETUAL_PUT = "perpetual-put"

    def build(self, *, numerical: bool, compact: bool) -> Study:
        # Import the selected study only when it is requested. Listing choices
        # must not load models or their optional numerical dependencies.
        from . import study

        match self:
            case _BuiltinModel.BLACK_SCHOLES:
                return study._black_scholes(numerical=numerical, compact=compact)
            case _BuiltinModel.PARTICIPATION_EXECUTION:
                return study._participation(numerical=numerical, compact=compact)
            case _BuiltinModel.MARKOWITZ:
                return study._markowitz(numerical=numerical, compact=compact)
            case _BuiltinModel.CIR_BOND:
                return study._cir(numerical=numerical, compact=compact)
            case _BuiltinModel.PERPETUAL_PUT:
                return study._perpetual(numerical=numerical, compact=compact)
            case _ as unexpected:
                assert_never(unexpected)

    def render(self, study: Study, plot: _Plot) -> None:
        from . import _plots

        match self:
            case _BuiltinModel.BLACK_SCHOLES:
                return _plots._black_scholes(study, plot)
            case _BuiltinModel.PARTICIPATION_EXECUTION:
                return _plots._participation(study, plot)
            case _BuiltinModel.MARKOWITZ:
                return _plots._markowitz(study, plot)
            case _BuiltinModel.CIR_BOND:
                return _plots._cir(study, plot)
            case _BuiltinModel.PERPETUAL_PUT:
                return _plots._perpetual(study, plot)
            case _ as unexpected:
                assert_never(unexpected)
