import pytest

from app import parse_indices, parse_slice_spec, GGUFParseError
import numpy as np


class TestParseIndicesBounds:
    def test_exact_upper_bound_raises(self):
        with pytest.raises(Exception):
            parse_indices("2,3", (2, 3))

    def test_zero_dimension_raises(self):
        with pytest.raises(Exception):
            parse_indices("0", (0,))

    def test_all_valid_boundary(self):
        assert parse_indices("0,0", (2, 3)) == (0, 0)
        assert parse_indices("1,2", (2, 3)) == (1, 2)

    def test_large_shape(self):
        assert parse_indices("999,999,999", (1000, 1000, 1000)) == (999, 999, 999)


class TestParseSliceSpecBounds:
    def test_negative_axis_raises(self):
        arr = np.arange(6, dtype=np.float32).reshape(2, 3)
        with pytest.raises(Exception):
            parse_slice_spec(arr, -1, 0)

    def test_negative_index_raises(self):
        arr = np.arange(6, dtype=np.float32).reshape(2, 3)
        with pytest.raises(Exception):
            parse_slice_spec(arr, 0, -1)

    def test_axis_equals_ndim_raises(self):
        arr = np.arange(6, dtype=np.float32).reshape(2, 3)
        with pytest.raises(Exception):
            parse_slice_spec(arr, 2, 0)

    def test_index_equals_size_raises(self):
        arr = np.arange(6, dtype=np.float32).reshape(2, 3)
        with pytest.raises(Exception):
            parse_slice_spec(arr, 0, 2)

    def test_valid_edge_cases(self):
        arr = np.arange(6, dtype=np.float32).reshape(2, 3)
        view, slicer = parse_slice_spec(arr, 0, 0)
        assert view.shape == (3,)
        view, slicer = parse_slice_spec(arr, 1, 2)
        assert view.shape == (2,)
