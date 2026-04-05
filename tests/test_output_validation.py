import os
import tempfile
import pytest
from pathlib import Path

from app import (
    validate_output_path,
    manifest_to_dict,
    patch_scalar,
    patch_transform,
    patch_slice,
    apply_batch,
    on_load,
    manifest_from_dict,
    batch_op_to_dict,
    BatchOperation,
)


class TestValidateOutputPath:
    def test_empty_path_raises(self, toy_path):
        with pytest.raises(Exception):
            validate_output_path(toy_path, "")

    def test_same_as_input_raises(self, toy_path):
        with pytest.raises(Exception, match="cannot be the same"):
            validate_output_path(toy_path, toy_path)

    def test_nonexistent_parent_creates_if_possible(self, toy_path):
        with tempfile.TemporaryDirectory() as tmpdir:
            new_dir = os.path.join(tmpdir, "new", "nested", "dir")
            out = os.path.join(new_dir, "output.gguf")
            resolved, warning = validate_output_path(toy_path, out)
            assert os.path.exists(new_dir)
            assert Path(resolved).resolve() == Path(out).resolve()

    def test_readonly_parent_raises(self, toy_path):
        with tempfile.TemporaryDirectory() as tmpdir:
            readonly_dir = os.path.join(tmpdir, "readonly")
            os.makedirs(readonly_dir, mode=0o444)
            try:
                out = os.path.join(readonly_dir, "output.gguf")
                with pytest.raises(Exception, match="not writable"):
                    validate_output_path(toy_path, out)
            finally:
                os.chmod(readonly_dir, 0o755)

    def test_existing_file_without_confirm_raises(self, toy_path, toy_manifest_dict):
        with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
            out = f.name
        try:
            with open(out, "wb") as f:
                f.write(b"test")

            with pytest.raises(Exception, match="already exists"):
                validate_output_path(toy_path, out, overwrite_confirm=False)

            resolved, warning = validate_output_path(
                toy_path, out, overwrite_confirm=True
            )
            assert "Overwriting" in warning
            assert Path(resolved).resolve() == Path(out).resolve()
        finally:
            if os.path.exists(out):
                os.unlink(out)


class TestPatchScalarWithValidation:
    def test_same_path_raises(self, toy_manifest_dict):
        with pytest.raises(Exception, match="cannot be the same"):
            patch_scalar(
                toy_manifest_dict,
                "tensor.test",
                "auto",
                "0,0",
                1.0,
                toy_manifest_dict["path"],
            )

    def test_existing_file_without_confirm_raises(self, toy_manifest_dict, toy_path):
        with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
            out = f.name
        try:
            with open(out, "wb") as f:
                f.write(b"test")

            with pytest.raises(Exception, match="already exists"):
                patch_scalar(
                    toy_manifest_dict,
                    "tensor.test",
                    "auto",
                    "0,0",
                    1.0,
                    out,
                )
        finally:
            if os.path.exists(out):
                os.unlink(out)

    def test_existing_file_with_confirm_succeeds(self, toy_manifest_dict, toy_path):
        with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
            out = f.name
        try:
            with open(out, "wb") as f:
                f.write(b"test")

            result = patch_scalar(
                toy_manifest_dict,
                "tensor.test",
                "auto",
                "0,0",
                1.0,
                out,
                overwrite_confirm=True,
            )
            assert "Overwriting" in result
            assert "Saved" in result
        finally:
            if os.path.exists(out):
                os.unlink(out)


class TestPatchTransformWithValidation:
    def test_same_path_raises(self, toy_manifest_dict):
        with pytest.raises(Exception, match="cannot be the same"):
            patch_transform(
                toy_manifest_dict,
                "tensor.test",
                "auto",
                2.0,
                0.0,
                None,
                None,
                toy_manifest_dict["path"],
            )


class TestPatchSliceWithValidation:
    def test_same_path_raises(self, toy_manifest_dict):
        with pytest.raises(Exception, match="cannot be the same"):
            patch_slice(
                toy_manifest_dict,
                "tensor.test",
                "auto",
                0,
                0,
                "set_constant",
                1.0,
                1.0,
                0.0,
                toy_manifest_dict["path"],
            )


class TestApplyBatchWithValidation:
    def test_same_path_raises(self, toy_manifest_dict):
        batch = [
            batch_op_to_dict(
                BatchOperation(
                    op_type="scalar",
                    tensor_name="tensor.test",
                    decode_as="auto",
                    parameters={"indices": "0,0", "new_value": 1.0},
                )
            )
        ]
        with pytest.raises(Exception, match="cannot be the same"):
            apply_batch(
                batch,
                toy_manifest_dict,
                toy_manifest_dict["path"],
            )

    def test_existing_file_without_confirm_raises(self, toy_manifest_dict, toy_path):
        with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
            out = f.name
        try:
            with open(out, "wb") as f:
                f.write(b"test")

            batch = [
                batch_op_to_dict(
                    BatchOperation(
                        op_type="scalar",
                        tensor_name="tensor.test",
                        decode_as="auto",
                        parameters={"indices": "0,0", "new_value": 1.0},
                    )
                )
            ]

            with pytest.raises(Exception, match="already exists"):
                apply_batch(batch, toy_manifest_dict, out)
        finally:
            if os.path.exists(out):
                os.unlink(out)
