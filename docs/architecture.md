# Architecture

## Overview

GGUF Workbench is a Python package with a Gradio UI for editing GGUF model weights.

## Directory Structure

```
gguf_workbench_v2_1/
├── app.py                  # Entry point (imports from gguf_workbench package)
├── gguf_workbench/         # Main package
│   ├── __init__.py         # Exports all public APIs
│   ├── constants.py        # GGUF/GGML constants
│   ├── parser.py           # GGUF file parsing
│   ├── tensor_ops.py       # Tensor decode/encode/transform
│   ├── validation.py       # Output path validation
│   ├── compare.py          # GGUF comparison
│   ├── batch.py            # Batch operation queue
│   └── app.py              # Gradio UI
├── tests/                  # Test suite
├── docs/                   # Documentation
└── transformed_models/     # Default output directory (gitignored)
```

## Module Responsibilities

### constants.py
All constants: GGUF magic number, supported versions, GGUF/GGML type IDs, editable type names.

### parser.py
GGUF parsing using memory-mapped I/O:
- `BinaryReader`: Binary stream reader
- `GGUFParser`: Main parser (context manager)
- `TensorInfo`, `GGUFManifest`: Data classes for tensor/model data

### tensor_ops.py
Tensor data operations:
- `decode_tensor()`: Read tensor from file
- `encode_tensor()`: Write tensor to file
- `transform_array()`: Apply scale+bias+clip
- `bf16_to_f32()`, `f32_to_bf16()`: Conversion utilities

### validation.py
- `validate_output_path()`: Validate output file path
- `default_output_path()`: Generate default output path in `transformed_models/`

### compare.py
- `compare_gguf()`: Compare two GGUF files, return diff summary

### batch.py
- `BatchOperation`: Dataclass for batch operations
- `batch_add_scalar()`, `batch_add_transform()`, `batch_add_slice()`: Add operations
- `apply_batch()`: Execute all batch operations

### app.py (in package)
- `build_app()`: Constructs Gradio interface
- All event handlers: `on_load`, `inspect_tensor`, `patch_scalar`, etc.

## Public API

Import from `app.py` for backwards compatibility:

```python
from app import (
    build_app,
    load_manifest,
    decode_tensor,
    patch_scalar,
    patch_transform,
    patch_slice,
    apply_batch,
    # ... etc
)
```

Or from the package:

```python
from gguf_workbench import build_app
```

## Running Tests

```bash
pytest
```

With coverage:

```bash
pytest --cov=gguf_workbench
```

## Supported Tensor Types

- ✅ F32, F16, BF16 (fully editable)
- ❌ Quantized types (Q4_K, Q5_K, Q6_K, IQ*, etc.)

## Key Design Decisions

1. **Memory-mapped I/O**: Uses `mmap` for efficient handling of large GGUF files
2. **Stateless operations**: Each patch reads from original, writes to new file
3. **Batch mode**: Multiple operations applied sequentially to a single copy
4. **No metadata editing**: Only tensor weights are editable (by design)