# GGUF Workbench V2.1

A Python + Gradio tool for surgically editing GGUF model weights. Make targeted edits to float tensors, queue multiple changes, and export a single modified model.

## Quick Start

```bash
# Install
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run
python app.py
```

Open `http://localhost:7860` in your browser.

## What It Does

- **Inspect** model structure: metadata, tensors, shapes, types
- **Edit** individual weights by index (scalar patching)
- **Transform** tensors using `new = old * scale + bias`
- **Patch** tensor slices along any axis
- **Batch** multiple edits and apply them at once

## Supported Types

- ✅ F32, F16, BF16 (float tensors)
- ❌ Quantized types (Q4_K, Q5_K, etc.) - intentionally not supported

## Output

All modified files are saved to `transformed_models/` directory.

## Documentation

See the `docs/` folder for detailed guides:

- `docs/installation.md` - Setup instructions
- `docs/usage.md` - Detailed usage guide
- `docs/troubleshooting.md` - Common issues and solutions
- `docs/architecture.md` - Code structure for developers

## License

MIT