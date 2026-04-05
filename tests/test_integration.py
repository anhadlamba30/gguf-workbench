import struct
import tempfile
import os
import numpy as np
import pytest

from app import (
    BinaryReader,
    GGUFParseError,
    GGML_TYPE_NAMES,
    GGUF_TYPE_UINT8,
    GGUF_TYPE_INT8,
    GGUF_TYPE_UINT16,
    GGUF_TYPE_INT16,
    GGUF_TYPE_UINT32,
    GGUF_TYPE_INT32,
    GGUF_TYPE_FLOAT32,
    GGUF_TYPE_BOOL,
    GGUF_TYPE_STRING,
    GGUF_TYPE_ARRAY,
    GGUF_TYPE_UINT64,
    GGUF_TYPE_INT64,
    GGUF_TYPE_FLOAT64,
    GGUF_MAGIC,
    GGML_TYPE_F32,
    GGML_TYPE_F16,
    GGML_TYPE_BF16,
    align_offset,
    inspect_tensor,
    on_load,
    patch_scalar,
    patch_transform,
    patch_slice,
    preview_transform,
    preview_slice_edit,
    decode_tensor,
    encode_tensor,
    manifest_from_dict,
)


class TestBinaryReader:
    def test_u32(self):
        br = BinaryReader(struct.pack("<I", 42))
        assert br.u32() == 42

    def test_u64(self):
        br = BinaryReader(struct.pack("<Q", 9999999999))
        assert br.u64() == 9999999999

    def test_i8(self):
        br = BinaryReader(struct.pack("<b", -42))
        assert br.i8() == -42

    def test_u8(self):
        br = BinaryReader(struct.pack("<B", 200))
        assert br.u8() == 200

    def test_i16(self):
        br = BinaryReader(struct.pack("<h", -1234))
        assert br.i16() == -1234

    def test_u16(self):
        br = BinaryReader(struct.pack("<H", 50000))
        assert br.u16() == 50000

    def test_i32(self):
        br = BinaryReader(struct.pack("<i", -999999))
        assert br.i32() == -999999

    def test_i64(self):
        br = BinaryReader(struct.pack("<q", -12345678901234))
        assert br.i64() == -12345678901234

    def test_f32(self):
        br = BinaryReader(struct.pack("<f", 3.14))
        assert abs(br.f32() - 3.14) < 1e-5

    def test_f64(self):
        br = BinaryReader(struct.pack("<d", 2.718281828))
        assert abs(br.f64() - 2.718281828) < 1e-10

    def test_boolean_true(self):
        br = BinaryReader(b"\x01")
        assert br.boolean() is True

    def test_boolean_false(self):
        br = BinaryReader(b"\x00")
        assert br.boolean() is False

    def test_string(self):
        data = struct.pack("<Q", 5) + b"hello"
        br = BinaryReader(data)
        assert br.string() == "hello"

    def test_tell(self):
        br = BinaryReader(b"\x00\x00\x00\x00")
        assert br.tell() == 0
        br.u32()
        assert br.tell() == 4

    def test_read_eof_raises(self):
        br = BinaryReader(b"\x00")
        with pytest.raises(GGUFParseError):
            br.read(5)

    def test_u32_eof_raises(self):
        br = BinaryReader(b"\x00\x00")
        with pytest.raises(GGUFParseError):
            br.u32()


class TestAlignOffset:
    def test_already_aligned(self):
        assert align_offset(32, 32) == 32

    def test_rounds_up(self):
        assert align_offset(33, 32) == 64

    def test_zero(self):
        assert align_offset(0, 32) == 0

    def test_alignment_1(self):
        assert align_offset(7, 1) == 7


class TestInspectTensor:
    def test_inspect_stats(self, toy_manifest_dict):
        stats, preview = inspect_tensor(toy_manifest_dict, "tensor.test", "auto", 32)
        assert "tensor.test" in stats
        assert "Shape" in stats
        assert "Elements" in stats
        assert "Min / Max" in stats
        assert len(preview) == 6

    def test_inspect_truncated(self, toy_manifest_dict):
        stats, preview = inspect_tensor(toy_manifest_dict, "tensor.test", "auto", 2)
        assert len(preview) == 2

    def test_inspect_unknown_tensor_raises(self, toy_manifest_dict):
        with pytest.raises(Exception):
            inspect_tensor(toy_manifest_dict, "nonexistent", "auto", 32)


class TestOnLoad:
    def test_no_path_raises(self):
        with pytest.raises(Exception):
            on_load("", None)

    def test_valid_load(self, toy_path):
        result = on_load(toy_path, None)
        manifest_dict, summary, meta_df, tensor_df = result[:4]
        assert manifest_dict["n_tensors"] == 1
        assert "toy.gguf" in summary
        assert len(meta_df) == 1
        assert len(tensor_df) == 1


class TestGGMLTypeNames:
    def test_known_types(self):
        assert GGML_TYPE_NAMES[GGML_TYPE_F32] == "F32"
        assert GGML_TYPE_NAMES[GGML_TYPE_F16] == "F16"
        assert GGML_TYPE_NAMES[GGML_TYPE_BF16] == "BF16"


class TestDecodeTensorErrors:
    def test_unsupported_type_raises(self, toy_tensor):
        bad_tensor = type(toy_tensor)(
            name=toy_tensor.name,
            shape=toy_tensor.shape,
            stored_dims=toy_tensor.stored_dims,
            ggml_type=99,
            ggml_type_name="TYPE_99",
            offset=toy_tensor.offset,
            abs_offset=toy_tensor.abs_offset,
            n_elements=toy_tensor.n_elements,
            n_bytes=toy_tensor.n_bytes,
            editable_kind="TYPE_99",
        )
        with pytest.raises(Exception):
            decode_tensor(toy_tensor.abs_offset, bad_tensor, "auto")


class TestEncodeTensorErrors:
    def test_unsupported_kind_raises(self, toy_path, toy_tensor):
        arr = decode_tensor(toy_path, toy_tensor, "F32")
        bad_tensor = type(toy_tensor)(
            name=toy_tensor.name,
            shape=toy_tensor.shape,
            stored_dims=toy_tensor.stored_dims,
            ggml_type=99,
            ggml_type_name="TYPE_99",
            offset=toy_tensor.offset,
            abs_offset=toy_tensor.abs_offset,
            n_elements=toy_tensor.n_elements,
            n_bytes=toy_tensor.n_bytes,
            editable_kind="TYPE_99",
        )
        with pytest.raises(Exception):
            encode_tensor(arr, bad_tensor, "auto")


class TestPatchScalarIntegration:
    def test_patch_and_verify(self, toy_manifest_dict, toy_path):
        with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
            out = f.name
        try:
            patch_scalar(
                toy_manifest_dict,
                "tensor.test",
                "auto",
                "1,2",
                42.0,
                out,
                overwrite_confirm=True,
            )
            manifest_dict, _, _, _ = on_load(out, None)[:4]
            m = manifest_from_dict(manifest_dict)
            stats, _ = inspect_tensor(manifest_dict, "tensor.test", "auto", 32)
            assert "42" in stats
        finally:
            os.unlink(out)


class TestPreviewTransformIntegration:
    def test_preview_matches_math(self, toy_manifest_dict):
        text, df = preview_transform(
            toy_manifest_dict,
            "tensor.test",
            "auto",
            2.0,
            0.0,
            None,
            None,
        )
        assert "Mean" in text
        assert len(df) > 0
        assert abs(df["after"].iloc[0] - df["before"].iloc[0] * 2) < 1e-5


class TestPreviewSliceEditIntegration:
    def test_preview_set_constant(self, toy_manifest_dict):
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
        assert 99.0 in df["after"].values


class TestPatchTransformIntegration:
    def test_transform_and_verify(self, toy_manifest_dict, toy_path):
        with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
            out = f.name
        try:
            patch_transform(
                toy_manifest_dict,
                "tensor.test",
                "auto",
                0.5,
                0.0,
                None,
                None,
                out,
                overwrite_confirm=True,
            )
            manifest_dict, _, _, _ = on_load(out, None)[:4]
            m = manifest_from_dict(manifest_dict)
            arr = decode_tensor(out, m.tensors[0], "auto")
            original = decode_tensor(toy_path, m.tensors[0], "auto")
            np.testing.assert_array_almost_equal(arr, original * 0.5)
        finally:
            os.unlink(out)


class TestPatchSliceIntegration:
    def test_slice_set_and_verify(self, toy_manifest_dict, toy_path):
        with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
            out = f.name
        try:
            patch_slice(
                toy_manifest_dict,
                "tensor.test",
                "auto",
                0,
                1,
                "set_constant",
                7.0,
                1.0,
                0.0,
                out,
                overwrite_confirm=True,
            )
            manifest_dict, _, _, _ = on_load(out, None)[:4]
            m = manifest_from_dict(manifest_dict)
            arr = decode_tensor(out, m.tensors[0], "auto")
            assert arr[1, 0] == pytest.approx(7.0)
            assert arr[1, 1] == pytest.approx(7.0)
            assert arr[1, 2] == pytest.approx(7.0)
        finally:
            os.unlink(out)
