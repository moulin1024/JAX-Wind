from __future__ import annotations

import numpy as np

from tools.prolong_pressure_driven_checkpoint import (
    prolong_cell_field,
    prolong_vertical_faces,
)


def test_cell_prolongation_preserves_constant_and_shape() -> None:
    source = np.full((2, 3, 4), 7.25, dtype=np.float32)

    result = prolong_cell_field(source, (4, 6, 8))

    assert result.shape == (4, 6, 8)
    assert result.dtype == np.float32
    np.testing.assert_array_equal(result, np.full(result.shape, 7.25))


def test_cell_prolongation_clamps_vertical_cell_centres() -> None:
    source = np.broadcast_to(
        np.asarray([1.0, 3.0], dtype=np.float32)[:, None, None],
        (2, 2, 2),
    )

    result = prolong_cell_field(source, (4, 4, 4))

    np.testing.assert_allclose(
        result[:, 0, 0],
        np.asarray([1.0, 1.5, 2.5, 3.0], dtype=np.float32),
    )


def test_vertical_face_prolongation_retains_both_walls() -> None:
    lower = np.zeros((2, 2), dtype=np.float32)
    upper = np.broadcast_to(
        np.asarray([1.0, 2.0], dtype=np.float32)[:, None, None],
        (2, 2, 2),
    )

    result_upper, result_lower = prolong_vertical_faces(
        upper,
        lower,
        (4, 4, 4),
    )

    assert result_upper.shape == (4, 4, 4)
    assert result_lower.shape == (4, 4)
    np.testing.assert_array_equal(result_lower, 0.0)
    np.testing.assert_allclose(
        result_upper[:, 0, 0],
        np.asarray([0.5, 1.0, 1.5, 2.0], dtype=np.float32),
    )
