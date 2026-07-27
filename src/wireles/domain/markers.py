"""Zero-storage static markers for initial quantities and phases."""

from __future__ import annotations


class Quantity:
    __slots__ = ()


class PressureCorrection(Quantity):
    __slots__ = ()


class PressureRhs(Quantity):
    __slots__ = ()


class XPressureGradient(Quantity):
    __slots__ = ()


class YPressureGradient(Quantity):
    __slots__ = ()


class VerticalPressureGradient(Quantity):
    __slots__ = ()


class XVelocity(Quantity):
    __slots__ = ()


class XVelocityTendency(Quantity):
    __slots__ = ()


class YVelocity(Quantity):
    __slots__ = ()


class YVelocityTendency(Quantity):
    __slots__ = ()


class VerticalVelocity(Quantity):
    __slots__ = ()


class VerticalVelocityTendency(Quantity):
    __slots__ = ()


class Divergence(Quantity):
    __slots__ = ()


class PotentialTemperaturePerturbation(Quantity):
    """Potential temperature minus one configured absolute reference, in K."""

    __slots__ = ()


class PotentialTemperatureTendency(Quantity):
    __slots__ = ()


class PassiveScalarConcentration(Quantity):
    """Passive mass concentration in canonical kg m-3 units."""

    __slots__ = ()


class PassiveScalarTendency(Quantity):
    __slots__ = ()


class MomentumLasdCoefficient(Quantity):
    __slots__ = ()


class MomentumLasdLm(Quantity):
    __slots__ = ()


class MomentumLasdMm(Quantity):
    __slots__ = ()


class MomentumLasdQn(Quantity):
    __slots__ = ()


class MomentumLasdNn(Quantity):
    __slots__ = ()


class LasdTrajectoryXVelocity(Quantity):
    __slots__ = ()


class LasdTrajectoryYVelocity(Quantity):
    __slots__ = ()


class LasdTrajectoryZVelocity(Quantity):
    __slots__ = ()


class ScalarLasdCoefficient(Quantity):
    __slots__ = ()


class ScalarLasdLm(Quantity):
    __slots__ = ()


class ScalarLasdMm(Quantity):
    __slots__ = ()


class ScalarLasdQn(Quantity):
    __slots__ = ()


class ScalarLasdNn(Quantity):
    __slots__ = ()


class Phase:
    __slots__ = ()


class Candidate(Phase):
    __slots__ = ()


class Accepted(Phase):
    """Accepted non-projected prognostic state."""

    __slots__ = ()


class Evaluated(Phase):
    __slots__ = ()


class Projected(Phase):
    __slots__ = ()
