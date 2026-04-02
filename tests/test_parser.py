import pytest
from pathlib import Path

from app import (
    GGUFParser, GGUFManifest, TensorInfo, GGUFParseError,
    load_manifest, filter_tensor_table, manifest_to_dict,
    manifest_from_dict, manifest_summary,
)


class TestGGUFParser:
    def test_parses_toy_gguf(self, toy_path: str):
        with GGUFParser(toy_path) as parser:
            manifest = parser.parse()
        assert manifest.version == 3
        assert manifest.n_tensors == 1
        assert manifest.n_kv == 1
        assert manifest.alignment == 32

    def test_file_size_matches(self, toy_path: str):
        with GGUFParser(toy_path) as parser:
            manifest = parser.parse()
        assert manifest.file_size == Path(toy_path).stat().st_size

    def test_tensor_data_start_aligned(self, toy_path: str):
        with GGUFParser(toy_path) as parser:
            manifest = parser.parse()
        assert manifest.tensor_data_start % manifest.alignment == 0

    def test_context_manager_closes(self, toy_path: str):
        parser = GGUFParser(toy_path)
        parser.parse()
        parser.close()
        assert parser._mm.closed
        assert parser._fd.closed

    def test_context_manager_with_statement(self, toy_path: str):
        with GGUFParser(toy_path) as parser:
            manifest = parser.parse()
        assert manifest.n_tensors == 1
        assert parser._mm.closed

    def test_invalid_path_raises(self):
        with pytest.raises(FileNotFoundError):
            with GGUFParser("/nonexistent/file.gguf") as parser:
                parser.parse()


class TestTensorInfo:
    def test_tensor_fields(self, toy_manifest: GGUFManifest):
        tensor = toy_manifest.tensors[0]
        assert tensor.name == "tensor.test"
        assert tensor.shape == (2, 3)
        assert tensor.stored_dims == (3, 2)
        assert tensor.ggml_type == 0
        assert tensor.ggml_type_name == "F32"
        assert tensor.n_elements == 6
        assert tensor.n_bytes == 24
        assert tensor.editable_kind == "F32"

    def test_to_row(self, toy_manifest: GGUFManifest):
        row = toy_manifest.tensors[0].to_row()
        assert row["name"] == "tensor.test"
        assert row["shape"] == [2, 3]
        assert row["elements"] == 6
        assert row["type"] == "F32"
        assert row["bytes"] == 24
        assert row["editable"] is True
        assert row["edit_kind"] == "F32"


class TestManifest:
    def test_tensor_map(self, toy_manifest: GGUFManifest):
        tmap = toy_manifest.tensor_map()
        assert "tensor.test" in tmap
        assert tmap["tensor.test"].name == "tensor.test"

    def test_editable_tensors(self, toy_manifest: GGUFManifest):
        editable = toy_manifest.editable_tensors()
        assert editable == ["tensor.test"]

    def test_manifest_summary(self, toy_manifest: GGUFManifest):
        summary = manifest_summary(toy_manifest)
        assert "toy.gguf" in summary
        assert "3" in summary
        assert "1" in summary

    def test_manifest_round_trip(self, toy_manifest: GGUFManifest):
        d = manifest_to_dict(toy_manifest)
        restored = manifest_from_dict(d)
        assert restored.version == toy_manifest.version
        assert restored.n_tensors == toy_manifest.n_tensors
        assert restored.tensors[0].name == toy_manifest.tensors[0].name
        assert restored.tensors[0].shape == toy_manifest.tensors[0].shape

    def test_manifest_from_dict_empty_raises(self):
        with pytest.raises(Exception):
            manifest_from_dict({})


class TestLoadManifest:
    def test_load_returns_four_values(self, toy_path: str):
        manifest, meta_df, tensor_df, choices = load_manifest(toy_path)
        assert isinstance(manifest, GGUFManifest)
        assert len(meta_df) == 1
        assert len(tensor_df) == 1
        assert choices == ["tensor.test"]

    def test_all_toy_files_load(self, toy_dir: Path):
        for f in ["toy.gguf", "toy_patched.gguf", "toy_slice.gguf", "toy_transformed.gguf"]:
            manifest, _, _, _ = load_manifest(str(toy_dir / f))
            assert manifest.n_tensors == 1

    def test_no_path_raisess(self):
        with pytest.raises(Exception):
            load_manifest("")


class TestFilterTensorTable:
    def test_no_filter_returns_all(self, toy_manifest_dict):
        df = filter_tensor_table(toy_manifest_dict, "", False)
        assert len(df) == 1

    def test_editable_only(self, toy_manifest_dict):
        df = filter_tensor_table(toy_manifest_dict, "", True)
        assert len(df) == 1

    def test_query_matches(self, toy_manifest_dict):
        df = filter_tensor_table(toy_manifest_dict, "test", False)
        assert len(df) == 1

    def test_query_no_match(self, toy_manifest_dict):
        df = filter_tensor_table(toy_manifest_dict, "nonexistent", False)
        assert len(df) == 0
