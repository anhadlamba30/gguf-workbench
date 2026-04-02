import pytest
import os
import tempfile

from app import (
    BatchOperation, batch_op_to_dict, batch_op_from_dict,
    batch_add_scalar, batch_add_transform, batch_add_slice,
    apply_batch, clear_batch, render_batch_queue,
    load_manifest, manifest_to_dict,
)


class TestBatchOperation:
    def test_scalar_label(self):
        op = BatchOperation(
            op_type="scalar", tensor_name="x", decode_as="auto",
            parameters={"indices": "0,1", "new_value": 3.14},
        )
        label = op.display_label()
        assert "Scalar" in label
        assert "x" in label
        assert "3.14" in label

    def test_transform_label(self):
        op = BatchOperation(
            op_type="transform", tensor_name="y", decode_as="auto",
            parameters={"scale": 2.0, "bias": 0.5, "clip_min": None, "clip_max": None},
        )
        label = op.display_label()
        assert "Transform" in label
        assert "y" in label

    def test_slice_label(self):
        op = BatchOperation(
            op_type="slice", tensor_name="z", decode_as="auto",
            parameters={"axis": 0, "index": 5, "mode": "set_constant", "value": 0.0, "scale": 1.0, "bias": 0.0},
        )
        label = op.display_label()
        assert "Slice" in label
        assert "axis=0" in label

    def test_unknown_label(self):
        op = BatchOperation(op_type="weird", tensor_name="w", decode_as="auto", parameters={})
        assert "Unknown" in op.display_label()

    def test_serialize_round_trip(self):
        op = BatchOperation(
            op_type="transform", tensor_name="x", decode_as="F32",
            parameters={"scale": 1.5, "bias": 0.0, "clip_min": None, "clip_max": None},
        )
        d = batch_op_to_dict(op)
        restored = batch_op_from_dict(d)
        assert restored.op_type == op.op_type
        assert restored.tensor_name == op.tensor_name
        assert restored.parameters == op.parameters


class TestBatchAdd:
    def test_add_scalar(self):
        batch, msg = batch_add_scalar([], "tensor.test", "auto", "0,0", 42.0)
        assert len(batch) == 1
        assert "Added" in msg

    def test_add_transform(self):
        batch, msg = batch_add_transform([], "tensor.test", "auto", 2.0, 0.0, None, None)
        assert len(batch) == 1
        assert "Added" in msg

    def test_add_slice(self):
        batch, msg = batch_add_slice([], "tensor.test", "auto", 0, 0, "set_constant", 0.0, 1.0, 0.0)
        assert len(batch) == 1
        assert "Added" in msg

    def test_add_scalar_requires_tensor(self):
        with pytest.raises(Exception):
            batch_add_scalar([], "", "auto", "0,0", 42.0)

    def test_add_transform_requires_tensor(self):
        with pytest.raises(Exception):
            batch_add_transform([], "", "auto", 1.0, 0.0, None, None)

    def test_add_slice_requires_tensor(self):
        with pytest.raises(Exception):
            batch_add_slice([], "", "auto", 0, 0, "set_constant", 0.0, 1.0, 0.0)

    def test_multiple_operations(self):
        batch, _ = batch_add_scalar([], "x", "auto", "0,0", 1.0)
        batch, _ = batch_add_transform(batch, "x", "auto", 2.0, 0.0, None, None)
        assert len(batch) == 2


class TestRenderBatchQueue:
    def test_empty_queue(self):
        md = render_batch_queue([])
        assert "Empty" in md

    def test_single_operation(self):
        batch, _ = batch_add_scalar([], "x", "auto", "0,0", 1.0)
        md = render_batch_queue(batch)
        assert "1 operation" in md

    def test_multiple_operations(self):
        batch, _ = batch_add_scalar([], "x", "auto", "0,0", 1.0)
        batch, _ = batch_add_scalar(batch, "x", "auto", "0,1", 2.0)
        md = render_batch_queue(batch)
        assert "2 operation" in md


class TestClearBatch:
    def test_clear_empties_queue(self):
        batch, _ = batch_add_scalar([], "x", "auto", "0,0", 1.0)
        cleared, msg = clear_batch(batch)
        assert cleared == []
        assert "Empty" in msg


class TestApplyBatch:
    def test_empty_batch_raises(self, toy_manifest_dict):
        with pytest.raises(Exception):
            apply_batch([], toy_manifest_dict, "/tmp/out.gguf")

    def test_no_output_raises(self, toy_manifest_dict):
        batch, _ = batch_add_scalar([], "tensor.test", "auto", "0,0", 1.0)
        with pytest.raises(Exception):
            apply_batch(batch, toy_manifest_dict, "")

    def test_apply_single_scalar(self, toy_manifest_dict, toy_path):
        batch, _ = batch_add_scalar([], "tensor.test", "auto", "0,0", 99.0)
        with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
            out_path = f.name
        try:
            result = apply_batch(batch, toy_manifest_dict, out_path)
            assert "Batch applied" in result
            assert "1 operation" in result

            manifest, _, _, _ = load_manifest(out_path)
            arr = manifest.tensors[0]
            from app import decode_tensor
            decoded = decode_tensor(out_path, arr, "auto")
            assert decoded[0, 0] == pytest.approx(99.0)
        finally:
            os.unlink(out_path)

    def test_apply_multiple_operations(self, toy_manifest_dict):
        batch, _ = batch_add_scalar([], "tensor.test", "auto", "0,0", 100.0)
        batch, _ = batch_add_transform(batch, "tensor.test", "auto", 1.0, 0.0, None, None)
        with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
            out_path = f.name
        try:
            result = apply_batch(batch, toy_manifest_dict, out_path)
            assert "2 operation" in result

            manifest, _, _, _ = load_manifest(out_path)
            from app import decode_tensor
            decoded = decode_tensor(out_path, manifest.tensors[0], "auto")
            assert decoded[0, 0] == pytest.approx(100.0)
        finally:
            os.unlink(out_path)

    def test_invalid_tensor_in_batch_raises(self, toy_manifest_dict):
        batch, _ = batch_add_scalar([], "nonexistent.tensor", "auto", "0,0", 1.0)
        with pytest.raises(Exception):
            apply_batch(batch, toy_manifest_dict, "/tmp/out.gguf")
