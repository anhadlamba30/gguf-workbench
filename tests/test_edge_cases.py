import struct
import numpy as np
import pytest

from app import (
    BinaryReader, GGUFParseError, GGUFParser,
    GGUF_TYPE_UINT8, GGUF_TYPE_INT8, GGUF_TYPE_UINT16, GGUF_TYPE_INT16,
    GGUF_TYPE_UINT32, GGUF_TYPE_INT32, GGUF_TYPE_FLOAT32, GGUF_TYPE_BOOL,
    GGUF_TYPE_STRING, GGUF_TYPE_ARRAY, GGUF_TYPE_UINT64, GGUF_TYPE_INT64,
    GGUF_TYPE_FLOAT64,
    GGUF_MAGIC, GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16,
    GGML_TYPE_NAMES, EDITABLE_TYPE_NAMES,
    align_offset, GGUFParseError,
    decode_tensor, encode_tensor,
    transform_array, parse_indices, parse_slice_spec,
    patch_scalar, patch_transform, patch_slice,
    preview_transform, preview_slice_edit,
    inspect_tensor, load_manifest, manifest_to_dict, manifest_from_dict,
    manifest_summary, default_output_path, filter_tensor_table,
    bf16_to_f32, f32_to_bf16, resolve_decode_kind,
    build_transform_preview,
    BatchOperation, batch_op_to_dict, batch_op_from_dict,
    batch_add_scalar, batch_add_transform, batch_add_slice,
    apply_batch, clear_batch, render_batch_queue,
    TensorInfo, GGUFManifest,
)


def _build_gguf_with_metadata(kv_pairs, tensor_name="t", shape=(2, 3), values=None):
    """Build a minimal GGUF v3 file with arbitrary metadata key-value pairs."""
    buf = bytearray()
    buf += GGUF_MAGIC
    buf += struct.pack("<I", 3)
    buf += struct.pack("<Q", 1)
    buf += struct.pack("<Q", len(kv_pairs))

    for key, value_type, value_bytes in kv_pairs:
        key_bytes = key.encode("utf-8")
        buf += struct.pack("<Q", len(key_bytes)) + key_bytes
        buf += struct.pack("<I", value_type)
        buf += value_bytes

    tensor_name_bytes = tensor_name.encode("utf-8")
    buf += struct.pack("<Q", len(tensor_name_bytes)) + tensor_name_bytes
    buf += struct.pack("<I", 2)
    buf += struct.pack("<Q", 3)
    buf += struct.pack("<Q", 2)
    buf += struct.pack("<I", GGML_TYPE_F32)
    buf += struct.pack("<Q", 0)

    alignment = 32
    header_end = len(buf)
    tensor_data_start = (header_end + alignment - 1) // alignment * alignment
    buf += b"\x00" * (tensor_data_start - header_end)

    if values is None:
        values = [0., 1., 2., 3., 4., 5.]
    for v in values:
        buf += struct.pack("<f", v)

    return bytes(buf)


def _write_gguf(tmp_path, kv_pairs, **kwargs):
    data = _build_gguf_with_metadata(kv_pairs, **kwargs)
    path = str(tmp_path / "test.gguf")
    with open(path, "wb") as f:
        f.write(data)
    return path


def _kv_uint8(v):
    return ("k", GGUF_TYPE_UINT8, struct.pack("<B", v))

def _kv_int8(v):
    return ("k", GGUF_TYPE_INT8, struct.pack("<b", v))

def _kv_uint16(v):
    return ("k", GGUF_TYPE_UINT16, struct.pack("<H", v))

def _kv_int16(v):
    return ("k", GGUF_TYPE_INT16, struct.pack("<h", v))

def _kv_uint32(v):
    return ("k", GGUF_TYPE_UINT32, struct.pack("<I", v))

def _kv_int32(v):
    return ("k", GGUF_TYPE_INT32, struct.pack("<i", v))

def _kv_float32(v):
    return ("k", GGUF_TYPE_FLOAT32, struct.pack("<f", v))

def _kv_bool(v):
    return ("k", GGUF_TYPE_BOOL, b"\x01" if v else b"\x00")

def _kv_string(v):
    vb = v.encode("utf-8")
    return ("k", GGUF_TYPE_STRING, struct.pack("<Q", len(vb)) + vb)

def _kv_array_uint32(items):
    inner = b""
    for x in items:
        inner += struct.pack("<I", x)
    return ("k", GGUF_TYPE_ARRAY, struct.pack("<I", GGUF_TYPE_UINT32) + struct.pack("<Q", len(items)) + inner)

def _kv_uint64(v):
    return ("k", GGUF_TYPE_UINT64, struct.pack("<Q", v))

def _kv_int64(v):
    return ("k", GGUF_TYPE_INT64, struct.pack("<q", v))

def _kv_float64(v):
    return ("k", GGUF_TYPE_FLOAT64, struct.pack("<d", v))


class TestReadAllValueTypes:
    def test_uint8(self, tmp_path):
        path = _write_gguf(tmp_path, [_kv_uint8(255)])
        m, _, _, _ = load_manifest(path)
        assert m.metadata["k"] == 255

    def test_int8(self, tmp_path):
        path = _write_gguf(tmp_path, [_kv_int8(-128)])
        m, _, _, _ = load_manifest(path)
        assert m.metadata["k"] == -128

    def test_uint16(self, tmp_path):
        path = _write_gguf(tmp_path, [_kv_uint16(65535)])
        m, _, _, _ = load_manifest(path)
        assert m.metadata["k"] == 65535

    def test_int16(self, tmp_path):
        path = _write_gguf(tmp_path, [_kv_int16(-32768)])
        m, _, _, _ = load_manifest(path)
        assert m.metadata["k"] == -32768

    def test_uint32(self, tmp_path):
        path = _write_gguf(tmp_path, [_kv_uint32(42)])
        m, _, _, _ = load_manifest(path)
        assert m.metadata["k"] == 42

    def test_int32(self, tmp_path):
        path = _write_gguf(tmp_path, [_kv_int32(-999)])
        m, _, _, _ = load_manifest(path)
        assert m.metadata["k"] == -999

    def test_float32(self, tmp_path):
        path = _write_gguf(tmp_path, [_kv_float32(3.14)])
        m, _, _, _ = load_manifest(path)
        assert abs(m.metadata["k"] - 3.14) < 1e-5

    def test_bool_true(self, tmp_path):
        path = _write_gguf(tmp_path, [_kv_bool(True)])
        m, _, _, _ = load_manifest(path)
        assert m.metadata["k"] is True

    def test_bool_false(self, tmp_path):
        path = _write_gguf(tmp_path, [_kv_bool(False)])
        m, _, _, _ = load_manifest(path)
        assert m.metadata["k"] is False

    def test_string(self, tmp_path):
        path = _write_gguf(tmp_path, [_kv_string("hello world")])
        m, _, _, _ = load_manifest(path)
        assert m.metadata["k"] == "hello world"

    def test_array_uint32(self, tmp_path):
        path = _write_gguf(tmp_path, [_kv_array_uint32([1, 2, 3])])
        m, _, _, _ = load_manifest(path)
        assert m.metadata["k"] == [1, 2, 3]

    def test_uint64(self, tmp_path):
        path = _write_gguf(tmp_path, [_kv_uint64(12345678901234)])
        m, _, _, _ = load_manifest(path)
        assert m.metadata["k"] == 12345678901234

    def test_int64(self, tmp_path):
        path = _write_gguf(tmp_path, [_kv_int64(-9876543210)])
        m, _, _, _ = load_manifest(path)
        assert m.metadata["k"] == -9876543210

    def test_float64(self, tmp_path):
        path = _write_gguf(tmp_path, [_kv_float64(2.718281828459)])
        m, _, _, _ = load_manifest(path)
        assert abs(m.metadata["k"] - 2.718281828459) < 1e-10


class TestInvalidValueType:
    def test_unknown_value_type_raises(self):
        br = BinaryReader(b"")
        parser = GGUFParser.__new__(GGUFParser)
        with pytest.raises(GGUFParseError):
            parser._read_value(br, 99)


class TestInferEditableKind:
    def test_f32_detected(self):
        t = TensorInfo("x", (2,), (2,), GGML_TYPE_F32, "F32", 0, 0, 2, 8, "UNKNOWN")
        parser = GGUFParser.__new__(GGUFParser)
        parser.path = None
        result = parser._infer_editable_kind(t)
        assert result == "F32"

    def test_f16_detected(self):
        t = TensorInfo("x", (2,), (2,), GGML_TYPE_F16, "F16", 0, 0, 2, 4, "UNKNOWN")
        parser = GGUFParser.__new__(GGUFParser)
        parser.path = None
        result = parser._infer_editable_kind(t)
        assert result == "F16"

    def test_bf16_detected(self):
        t = TensorInfo("x", (2,), (2,), GGML_TYPE_BF16, "BF16", 0, 0, 2, 4, "UNKNOWN")
        parser = GGUFParser.__new__(GGUFParser)
        parser.path = None
        result = parser._infer_editable_kind(t)
        assert result == "BF16"

    def test_fallback_by_size_4(self):
        t = TensorInfo("x", (2,), (2,), 99, "UNKNOWN", 0, 0, 2, 8, "UNKNOWN")
        parser = GGUFParser.__new__(GGUFParser)
        parser.path = None
        result = parser._infer_editable_kind(t)
        assert result == "F32"

    def test_fallback_by_size_2(self):
        t = TensorInfo("x", (2,), (2,), 99, "UNKNOWN", 0, 0, 2, 4, "UNKNOWN")
        parser = GGUFParser.__new__(GGUFParser)
        parser.path = None
        result = parser._infer_editable_kind(t)
        assert result == "F16_OR_BF16"

    def test_unknown_type_returns_name(self):
        t = TensorInfo("x", (2,), (2,), 50, "TYPE_50", 0, 0, 2, 16, "UNKNOWN")
        parser = GGUFParser.__new__(GGUFParser)
        parser.path = None
        result = parser._infer_editable_kind(t)
        assert result == "TYPE_50"


class TestEncodeTensorSizeMismatch:
    def test_byte_length_mismatch_raises(self, toy_tensor):
        arr = np.array([1.0], dtype=np.float32)
        with pytest.raises(Exception):
            encode_tensor(arr, toy_tensor, "F32")


class TestTransformArrayEdgeCases:
    def test_empty_array(self):
        arr = np.array([], dtype=np.float32)
        result = transform_array(arr, 2.0, 1.0, None, None)
        assert len(result) == 0

    def test_single_element(self):
        arr = np.array([5.0], dtype=np.float32)
        result = transform_array(arr, 0.5, 1.0, None, None)
        assert result[0] == pytest.approx(3.5)


class TestBuildTransformPreviewEdgeCases:
    def test_single_element_preview(self):
        before = np.array([1.0], dtype=np.float32)
        after = np.array([2.0], dtype=np.float32)
        text, df = build_transform_preview(before, after)
        assert len(df) == 1


class TestBatchApplyErrorPaths:
    def test_unknown_op_type_raises(self, toy_manifest_dict):
        op = BatchOperation(op_type="bogus", tensor_name="tensor.test", decode_as="auto", parameters={})
        batch = [batch_op_to_dict(op)]
        with pytest.raises(Exception) as exc_info:
            apply_batch(batch, toy_manifest_dict, "/tmp/out.gguf")
        assert "bogus" in str(exc_info.value)

    def test_op_failure_propagates(self, toy_manifest_dict):
        batch, _ = batch_add_scalar([], "tensor.test", "auto", "99,99", 1.0)
        with pytest.raises(Exception) as exc_info:
            apply_batch(batch, toy_manifest_dict, "/tmp/out.gguf")
        assert "Operation 1" in str(exc_info.value)


class TestManifestFromDictNone:
    def test_none_raises(self):
        with pytest.raises(Exception):
            manifest_from_dict(None)


class TestFilterTensorTableEdgeCases:
    def test_none_query(self, toy_manifest_dict):
        df = filter_tensor_table(toy_manifest_dict, None, False)
        assert len(df) == 1

    def test_whitespace_query(self, toy_manifest_dict):
        df = filter_tensor_table(toy_manifest_dict, "   ", False)
        assert len(df) == 1


class TestParseIndicesEdgeCases:
    def test_empty_string_raises(self):
        with pytest.raises(Exception):
            parse_indices("", (2, 3))

    def test_spaces_only_raises(self):
        with pytest.raises(Exception):
            parse_indices("  ,  ", (2, 3))


class TestParseSliceSpecEdgeCases:
    def test_1d_tensor(self):
        arr = np.arange(10, dtype=np.float32)
        view, slicer = parse_slice_spec(arr, 0, 5)
        assert view.shape == ()
        assert view == 5.0


class TestDefaultOutputPathEdgeCases:
    def test_nested_path(self):
        assert default_output_path("/a/b/c.gguf") == "/a/b/c.patched.gguf"

    def test_dot_in_name(self):
        assert default_output_path("/models/my.model.gguf") == "/models/my.model.patched.gguf"


class TestBatchOperationFromDict:
    def test_round_trip_scalar(self):
        op = BatchOperation("scalar", "x", "auto", {"indices": "0", "new_value": 1.0})
        d = batch_op_to_dict(op)
        r = batch_op_from_dict(d)
        assert r.op_type == "scalar"
        assert r.parameters["new_value"] == 1.0

    def test_round_trip_transform(self):
        op = BatchOperation("transform", "x", "F32", {"scale": 2.0, "bias": 0.0, "clip_min": None, "clip_max": None})
        d = batch_op_to_dict(op)
        r = batch_op_from_dict(d)
        assert r.op_type == "transform"
        assert r.parameters["scale"] == 2.0

    def test_round_trip_slice(self):
        op = BatchOperation("slice", "x", "auto", {"axis": 0, "index": 1, "mode": "set_constant", "value": 0.0, "scale": 1.0, "bias": 0.0})
        d = batch_op_to_dict(op)
        r = batch_op_from_dict(d)
        assert r.op_type == "slice"
        assert r.parameters["axis"] == 0


class TestClearBatch:
    def test_clear_empty_batch(self):
        cleared, msg = clear_batch([])
        assert cleared == []
        assert "Empty" in msg


class TestRenderBatchQueueEmpty:
    def test_none_like_empty(self):
        md = render_batch_queue([])
        assert "Empty" in md


class TestGGMLTypeNamesComplete:
    def test_all_known_types(self):
        assert GGML_TYPE_NAMES[0] == "F32"
        assert GGML_TYPE_NAMES[1] == "F16"
        assert GGML_TYPE_NAMES[30] == "BF16"

    def test_unknown_type_default(self):
        assert GGML_TYPE_NAMES.get(99) is None


class TestEditableTypeNames:
    def test_f32_editable(self):
        assert "F32" in EDITABLE_TYPE_NAMES

    def test_f16_editable(self):
        assert "F16" in EDITABLE_TYPE_NAMES

    def test_bf16_editable(self):
        assert "BF16" in EDITABLE_TYPE_NAMES

    def test_f16_or_bf16_editable(self):
        assert "F16_OR_BF16" in EDITABLE_TYPE_NAMES
