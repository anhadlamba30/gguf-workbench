# Usage Guide

## Loading a Model

1. Click "Upload GGUF file" and select your `.gguf` file
2. Click "Load GGUF"
3. The model metadata and tensor list will appear

## UI Tabs

### Metadata + Tensors

View model structure:
- **Metadata table**: All key-value pairs (model name, tokenizer info, etc.)
- **Tensor list**: Tensors with shape, type, element count, editability
- **Filter**: Search by tensor name (case-insensitive, substring match)
- **Editable only**: Show only float-like tensors (F32, F16, BF16)

### Inspect

Deep-dive into a single tensor:
1. Select a tensor from the dropdown
2. Choose decode method ("auto" is usually fine)
3. Set preview size
4. Click "Inspect"

Output shows: shape, type, element count, min/max/mean/std stats, flattened preview table.

### Patch Scalar

Edit exactly one number:
1. Select the tensor
2. Enter indices as comma-separated (e.g., `0,1,2`)
3. Enter the new value
4. Click "Apply now" (immediate) or "Add to batch" (queue for later)

### Transform Whole Tensor

Apply `new = old * scale + bias` to every value:
1. Select the tensor
2. Set **Scale** (multiply all values)
3. Set **Bias** (add to all values)
4. Optionally set **Clip min** / **Clip max** to cap values
5. Click "Preview transform" to see before/after stats
6. Click "Apply now" or "Add to batch"

### Patch Slice

Edit one slice along a specific axis:
1. Select the tensor
2. Choose **Axis** (0, 1, 2, etc.)
3. Choose **Index** along that axis
4. Choose **Mode**:
   - `set_constant`: Replace slice with one value
   - `scale_and_bias`: Apply `slice = slice * scale + bias`
5. Set parameters and click "Apply now" or "Add to batch"

### Compare

Compare two GGUF files to see what changed:
1. Upload original GGUF
2. Upload patched GGUF
3. Click "Compare"

### Batch Manager

Queue multiple edits and apply them together:
1. Use Patch scalar, Transform, or Patch slice tabs
2. For each edit, click "Add to batch"
3. Go to Batch Manager tab
4. Review queued operations
5. Enter output path
6. Click "Apply batch"

## Batch vs Immediate Mode

**Use batch when you:**
- Need 2+ edits to the same model
- Want a single output file
- Want to review all changes before applying

**Use "Apply now" when you:**
- Making a single quick edit
- Want instant feedback

## Tips

- Always use "Inspect" first to understand tensor values
- Use "Preview" for Transform and Slice operations
- Start with small scale/bias values (e.g., 0.95–1.05)
- Keep original GGUF files - this tool doesn't include undo
- Test with small edits first before complex batch operations