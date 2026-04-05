import numpy as np
import pytest

from app import (
    transform_array,
    preview_transform,
    build_transform_preview,
    parse_indices,
    parse_slice_spec,
    patch_scalar,
    patch_transform,
    patch_slice,
    preview_slice_edit,
    inspect_tensor,
    default_output_path,
)


class TestTransformArray:
    def test_identity(self):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = transform_array(arr, 1.0, 0.0, None, None)
        np.testing.assert_array_almost_equal(result, arr)

    def test_scale_only(self):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = transform_array(arr, 2.0, 0.0, None, None)
        np.testing.assert_array_almost_equal(result, [2.0, 4.0, 6.0])

    def test_bias_only(self):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = transform_array(arr, 1.0, 0.5, None, None)
        np.testing.assert_array_almost_equal(result, [1.5, 2.5, 3.5])

    def test_scale_and_bias(self):
        arr = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        result = transform_array(arr, 2.0, 1.0, None, None)
        expected = np.array([1.0, 3.0, 5.0, 7.0, 9.0, 11.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)

    def test_clip_min(self):
        arr = np.array([-2.0, 0.0, 3.0], dtype=np.float32)
        result = transform_array(arr, 1.0, 0.0, 0.0, None)
        np.testing.assert_array_almost_equal(result, [0.0, 0.0, 3.0])

    def test_clip_max(self):
        arr = np.array([-2.0, 0.0, 3.0], dtype=np.float32)
        result = transform_array(arr, 1.0, 0.0, None, 1.0)
        np.testing.assert_array_almost_equal(result, [-2.0, 0.0, 1.0])

    def test_clip_both(self):
        arr = np.array([-5.0, 0.0, 5.0], dtype=np.float32)
        result = transform_array(arr, 1.0, 0.0, -1.0, 1.0)
        np.testing.assert_array_almost_equal(result, [-1.0, 0.0, 1.0])

    def test_nan_clip_disabled(self):
        arr = np.array([-5.0, 0.0, 5.0], dtype=np.float32)
        result = transform_array(arr, 1.0, 0.0, float("nan"), float("nan"))
        np.testing.assert_array_almost_equal(result, arr)

    def test_multidim(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        result = transform_array(arr, 0.5, 0.0, None, None)
        np.testing.assert_array_almost_equal(result, [[0.5, 1.0], [1.5, 2.0]])


class TestBuildTransformPreview:
    def test_preview_output(self):
        before = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        after = np.array([2.0, 4.0, 6.0], dtype=np.float32)
        text, df = build_transform_preview(before, after, preview_n=16)
        assert "Mean" in text
        assert len(df) == 3
        assert list(df.columns) == ["flat_index", "before", "after", "delta"]

    def test_preview_truncates(self):
        before = np.arange(100, dtype=np.float32)
        after = before * 2
        text, df = build_transform_preview(before, after, preview_n=5)
        assert len(df) == 5


class TestParseIndices:
    def test_valid_indices(self):
        assert parse_indices("0,1,2", (3, 4, 5)) == (0, 1, 2)

    def test_wrong_count_raises(self):
        with pytest.raises(Exception):
            parse_indices("0,1", (3, 4, 5))

    def test_out_of_bounds_raises(self):
        with pytest.raises(Exception):
            parse_indices("0,10,2", (3, 4, 5))

    def test_negative_raises(self):
        with pytest.raises(Exception):
            parse_indices("-1,1,2", (3, 4, 5))

    def test_single_dim(self):
        assert parse_indices("5", (10,)) == (5,)


class TestParseSliceSpec:
    def test_axis_0(self):
        arr = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        slice_view, slicer = parse_slice_spec(arr, 0, 1)
        assert slice_view.shape == (3, 4)
        assert slicer == (1, slice(None), slice(None))

    def test_axis_1(self):
        arr = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        slice_view, slicer = parse_slice_spec(arr, 1, 2)
        assert slice_view.shape == (2, 4)
        assert slicer == (slice(None), 2, slice(None))

    def test_axis_out_of_bounds_raises(self):
        arr = np.arange(6, dtype=np.float32).reshape(2, 3)
        with pytest.raises(Exception):
            parse_slice_spec(arr, 2, 0)

    def test_index_out_of_bounds_raises(self):
        arr = np.arange(6, dtype=np.float32).reshape(2, 3)
        with pytest.raises(Exception):
            parse_slice_spec(arr, 0, 5)


class TestPatchScalar:
    def test_patch_single_value(self, toy_manifest_dict):
        result = patch_scalar(
            toy_manifest_dict,
            "tensor.test",
            "auto",
            "0,0",
            42.0,
            "/tmp/test_scalar.gguf",
            overwrite_confirm=True,
        )
        assert "42" in result
        assert "tensor.test" in result

    def test_patch_requires_tensor_name(self, toy_manifest_dict):
        with pytest.raises(Exception):
            patch_scalar(
                toy_manifest_dict,
                "",
                "auto",
                "0,0",
                42.0,
                "/tmp/test.gguf",
                overwrite_confirm=True,
            )


class TestPatchTransform:
    def test_transform_saves_file(self, toy_manifest_dict):
        result = patch_transform(
            toy_manifest_dict,
            "tensor.test",
            "auto",
            2.0,
            1.0,
            None,
            None,
            "/tmp/test_transform.gguf",
            overwrite_confirm=True,
        )
        assert "Saved" in result
        assert "test_transform.gguf" in result


class TestPatchSlice:
    def test_set_constant(self, toy_manifest_dict):
        result = patch_slice(
            toy_manifest_dict,
            "tensor.test",
            "auto",
            0,
            1,
            "set_constant",
            7.0,
            1.0,
            0.0,
            "/tmp/test_slice.gguf",
            overwrite_confirm=True,
        )
        assert "axis=0" in result
        assert "index=1" in result

    def test_scale_and_bias(self, toy_manifest_dict):
        result = patch_slice(
            toy_manifest_dict,
            "tensor.test",
            "auto",
            0,
            0,
            "scale_and_bias",
            0.0,
            2.0,
            1.0,
            "/tmp/test_slice_sb.gguf",
            overwrite_confirm=True,
        )
        assert "Saved" in result


class TestPreviewSliceEdit:
    def test_preview_returns_stats(self, toy_manifest_dict):
        text, df = preview_slice_edit(
            toy_manifest_dict,
            "tensor.test",
            "auto",
            0,
            0,
            "set_constant",
            99.0,
            1.0,
            0.0,
        )
        assert "axis=0" in text
        assert len(df) > 0


class TestDefaultOutputPath:
    def test_default_suffix(self):
        result = default_output_path("/models/foo.gguf")
        assert result.endswith("foo.patched.gguf")

    def test_custom_suffix(self):
        result = default_output_path("/models/foo.gguf", ".out.gguf")
        assert result.endswith("foo.out.gguf")
