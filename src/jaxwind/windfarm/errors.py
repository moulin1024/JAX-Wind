"""Errors raised while importing OpenFAST-format turbine data."""


class OpenFASTInputError(ValueError):
    """An OpenFAST deck cannot be represented by the rigid JAX-Wind model."""
