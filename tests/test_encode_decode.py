import numpy as np
import pytest

from app import (
    decode_tensor, encode_tensor, bf16_to_f32, f32_to_bf16,
    resolve_decode_kind, GGUFManifest, TensorInfo,
)


class TestDecodeTensor:
    def test_decode_toy_f32(self, toy_path, toy_tensor):
        arr = decode_tensor(toy_path, toy_tensor, "auto")
        assert arr.shape == (2, 3)
        assert arr.dtype == np.float32
        expected = np.array([[0., 1., 2.], [3., 4., 5.]], dtype=np.float32)
        np.testing.assert_array_equal(arr, expected)

    def test_decode_explicit_f32(self, toy_path, toy_tensor):
        arr = decode_tensor(toy_path, toy_tensor, "F32")
        assert arr.shape == (2, 3)

    def test_decode_patched(self, toy_patched_path, toy_tensor):
        arr = decode_tensor(toy_patched_path, toy_tensor, "auto")
        flat = arr.reshape(-1)
        assert flat[-1] == pytest.approx(99.0)

    def test_decode_slice(self, toy_slice_path, toy_tensor):
        arr = decode_tensor(toy_slice_path, toy_tensor, "auto")
        flat = arr.reshape(-1)
        np.testing.assert_array_almost_equal(flat[3:], [7., 7., 7.])

    def test_decode_transformed(self, toy_transformed_path, toy_tensor):
        arr = decode_tensor(toy_transformed_path, toy_tensor, "auto")
        flat = arr.reshape(-1)
        expected = np.array([1., 3., 5., 7., 9., 11.], dtype=np.float32)
        np.testing.assert_array_almost_equal(flat, expected)


class TestEncodeDecodeRoundTrip:
    def test_f32_round_trip(self, toy_path, toy_tensor):
        arr = decode_tensor(toy_path, toy_tensor, "F32")
        data = encode_tensor(arr, toy_tensor, "F32")
        assert len(data) == toy_tensor.n_bytes
        recovered = np.frombuffer(data, dtype="<f4")
        np.testing.assert_array_almost_equal(recovered, arr.reshape(-1))

    def test_f16_round_trip(self, toy_path, toy_tensor):
        arr = decode_tensor(toy_path, toy_tensor, "F32")
        f16_tensor = TensorInfo(
            name=toy_tensor.name,
            shape=toy_tensor.shape,
            stored_dims=toy_tensor.stored_dims,
            ggml_type=1,
            ggml_type_name="F16",
            offset=toy_tensor.offset,
            abs_offset=toy_tensor.abs_offset,
            n_elements=toy_tensor.n_elements,
            n_bytes=toy_tensor.n_elements * 2,
            editable_kind="F16",
        )
        data = encode_tensor(arr, f16_tensor, "F16")
        assert len(data) == f16_tensor.n_bytes
        recovered_f16 = np.frombuffer(data, dtype="<f2")
        np.testing.assert_array_almost_equal(recovered_f16.astype(np.float32), arr.reshape(-1), decimal=3)

    def test_bf16_round_trip(self, toy_path, toy_tensor):
        arr = decode_tensor(toy_path, toy_tensor, "F32")
        bf16_tensor = TensorInfo(
            name=toy_tensor.name,
            shape=toy_tensor.shape,
            stored_dims=toy_tensor.stored_dims,
            ggml_type=30,
            ggml_type_name="BF16",
            offset=toy_tensor.offset,
            abs_offset=toy_tensor.abs_offset,
            n_elements=toy_tensor.n_elements,
            n_bytes=toy_tensor.n_elements * 2,
            editable_kind="BF16",
        )
        data = encode_tensor(arr, bf16_tensor, "BF16")
        assert len(data) == bf16_tensor.n_bytes
        recovered_u16 = np.frombuffer(data, dtype="<u2")
        recovered_f32 = bf16_to_f32(recovered_u16)
        np.testing.assert_array_almost_equal(recovered_f32, arr.reshape(-1), decimal=2)


class TestBf16Conversion:
    def test_bf16_to_f32_basic(self):
        u16 = np.array([0x0000, 0x3F80, 0x4000, 0x4040], dtype=np.uint16)
        f32 = bf16_to_f32(u16)
        assert f32.dtype == np.float32
        assert len(f32) == 4

    def test_f32_to_bf16_to_f32(self):
        original = np.array([1.0, 2.0, 0.5, -0.25], dtype=np.float32)
        u16 = f32_to_bf16(original)
        recovered = bf16_to_f32(u16)
        np.testing.assert_array_almost_equal(original, recovered, decimal=2)


class TestResolveDecodeKind:
    def test_auto_f32(self, toy_tensor):
        assert resolve_decode_kind(toy_tensor, "auto") == "F32"

    def test_explicit_override(self, toy_tensor):
        assert resolve_decode_kind(toy_tensor, "F16") == "F16"

    def test_f16_or_bf16_defaults_f16(self):
        fake = TensorInfo(
            name="x", shape=(2,), stored_dims=(2,), ggml_type=99,
            ggml_type_name="UNKNOWN", offset=0, abs_offset=0,
            n_elements=2, n_bytes=4, editable_kind="F16_OR_BF16",
        )
        assert resolve_decode_kind(fake, "auto") == "F16"
