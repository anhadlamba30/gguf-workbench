# GGUF Workbench Repository Structure

## Overview

GGUF Workbench V2.1 is a Python + Gradio tool for surgically editing GGUF model weights. It allows targeted edits to float tensors, queuing multiple changes, and exporting a single modified model.

## Directory Structure

```
gguf_workbench_v2_1/
├── .tickets/              # Feature tickets and issue tracking
│   ├── manifest.json      # Ticket status tracker
│   └── *.md               # Individual ticket files
├── .git/                  # Git repository
├── tests/                 # Test suite
│   ├── conftest.py       # Pytest configuration and fixtures
│   ├── test_*.py         # Test modules
├── gguf_workbench/        # Main package (created during refactor)
│   ├── __init__.py       # Package exports and version
│   ├── constants.py      # All GGUF/GGML constants
│   ├── parser.py         # GGUF file parsing (BinaryReader, GGUFParser)
│   ├── tensor_ops.py     # Tensor decode/encode/transform operations
│   ├── validation.py     # Output path validation utilities
│   ├── compare.py        # GGUF comparison/diff logic
│   ├── batch.py          # Batch operation queue and execution
│   └── app.py            # Gradio UI (build_app function)
├── app.py                # Entry point (imports from gguf_workbench package)
├── requirements.txt      # Python dependencies
├── pyproject.toml        # Project configuration
├── README.md             # Main documentation
└── LICENSE               # MIT License
```

## Module Responsibilities

### `gguf_workbench/constants.py`
All constants used throughout the codebase:
- GGUF magic number, supported versions
- GGUF metadata types (GGUF_TYPE_*)
- GGML tensor types (GGML_TYPE_*)
- Editable type names

### `gguf_workbench/parser.py`
GGUF file parsing and manifest handling:
- `BinaryReader`: Binary stream reader for GGUF files
- `GGUFParser`: Main parser using mmap for memory efficiency
- `TensorInfo`: Dataclass for tensor metadata
- `GGUFManifest`: Dataclass for full model manifest
- `load_gguf()`, `load_manifest()`, `manifest_to_dict()`, `manifest_from_dict()`

### `gguf_workbench/tensor_ops.py`
Tensor data operations:
- `bf16_to_f32()`, `f32_to_bf16()`: BF16 conversion utilities
- `resolve_decode_kind()`: Determine tensor encoding
- `decode_tensor()`: Read tensor data from file
- `encode_tensor()`: Write tensor data to file
- `transform_array()`: Apply scale+bias+clip to arrays
- `build_transform_preview()`: Generate before/after stats
- `parse_indices()`, `parse_slice_spec()`: Index parsing

### `gguf_workbench/validation.py`
Output path validation:
- `validate_output_path()`: Validate output file paths
- `default_output_path()`: Generate default output filename

### `gguf_workbench/compare.py`
GGUF file comparison:
- `compare_gguf()`: Compare two GGUF files, return diff summary

### `gguf_workbench/batch.py`
Batch operation management:
- `BatchOperation`: Dataclass for batch operations
- `batch_add_scalar()`, `batch_add_transform()`, `batch_add_slice()`: Add ops
- `render_batch_queue()`: Display batch queue
- `apply_batch()`: Execute all batch operations
- `clear_batch()`: Clear batch queue
- `write_tensor_patch()`: Write modified tensor to file

### `gguf_workbench/app.py`
Gradio UI construction:
- `build_app()`: Constructs the full Gradio interface
- All tab handlers: on_load, inspect_tensor, patch_scalar, etc.

## Entry Points

### Running the App
```bash
# Using the entry point (recommended)
python app.py

# Or directly from the package
python -m gguf_workbench.app
```

### Running Tests
```bash
# All tests
pytest

# Specific test file
pytest tests/test_parser.py

# With coverage
pytest --cov=gguf_workbench
```

## Dependency Graph

```
app.py (entry point)
  └── gguf_workbench/
       ├── __init__.py (exports)
       ├── app.py (UI)
       │    ├── constants.py
       │    ├── parser.py
       │    ├── tensor_ops.py
       │    ├── validation.py
       │    ├── compare.py
       │    └── batch.py
       ├── parser.py → constants.py
       ├── tensor_ops.py → parser.py, constants.py
       ├── validation.py (standalone)
       ├── compare.py → parser.py, tensor_ops.py
       └── batch.py → parser.py, tensor_ops.py, validation.py
```

## Quick Reference

| Task | Module/Function |
|------|-----------------|
| Parse GGUF file | `gguf_workbench.load_gguf()` |
| Read tensor data | `gguf_workbench.decode_tensor()` |
| Write tensor data | `gguf_workbench.encode_tensor()` |
| Transform tensor | `gguf_workbench.transform_array()` |
| Validate paths | `gguf_workbench.validate_output_path()` |
| Compare files | `gguf_workbench.compare_gguf()` |
| Run batch ops | `gguf_workbench.apply_batch()` |
| Build UI | `gguf_workbench.build_app()` |

## Supported Tensor Types

- **F32**: 32-bit float (fully editable)
- **F16**: 16-bit half precision (fully editable)
- **BF16**: bfloat16 (fully editable)

Quantized types (Q4_K, Q5_K, Q6_K, etc.) are **not supported** by design.

## Architecture Notes

1. **Memory-mapped I/O**: Uses mmap for efficient handling of large GGUF files
2. **Stateless operations**: Each patch operation reads from original, writes to new file
3. **Batch mode**: Multiple operations applied in sequence to a single copy
4. **Gradio state**: Uses gr.State for manifest and batch queue persistence