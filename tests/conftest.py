import pytest
from pathlib import Path

from app import load_manifest, decode_tensor, manifest_to_dict, GGUFManifest, TensorInfo

TOY_DIR = Path(__file__).resolve().parent.parent


@pytest.fixture
def toy_dir() -> Path:
    return TOY_DIR


@pytest.fixture
def toy_path(toy_dir: Path) -> str:
    return str(toy_dir / "toy.gguf")


@pytest.fixture
def toy_patched_path(toy_dir: Path) -> str:
    return str(toy_dir / "toy_patched.gguf")


@pytest.fixture
def toy_slice_path(toy_dir: Path) -> str:
    return str(toy_dir / "toy_slice.gguf")


@pytest.fixture
def toy_transformed_path(toy_dir: Path) -> str:
    return str(toy_dir / "toy_transformed.gguf")


@pytest.fixture
def toy_manifest(toy_path: str):
    manifest, _, _, _ = load_manifest(toy_path)
    return manifest


@pytest.fixture
def toy_manifest_dict(toy_path: str):
    manifest, _, _, _ = load_manifest(toy_path)
    return manifest_to_dict(manifest)


@pytest.fixture
def toy_tensor(toy_manifest: GGUFManifest) -> TensorInfo:
    return toy_manifest.tensors[0]
