# Troubleshooting

## "Tensor not found"

**Problem:** Can't find a tensor you're looking for.

**Solution:**
1. Go to **Metadata + Tensors**
2. Use the filter box to search
3. Tensor names are case-sensitive
4. Check **Editable only** to see what's actually editable

---

## "Indices out of bounds"

**Problem:** Indices like `10,5,2` but tensor is smaller.

**Solution:**
1. Go to **Inspect** to check tensor shape
2. Verify indices are within bounds
3. Example: If shape is `[100, 50, 10]`, valid indices are `[0-99, 0-49, 0-9]`

---

## "Tensor is not editable"

**Problem:** Selected a quantized tensor (e.g., Q4_K).

**Solution:**
1. This tool only edits float tensors by design
2. Look for tensors with types F32, F16, or BF16
3. Dequantize the model elsewhere if you need to edit quantized weights

---

## Output file is corrupted

**Problem:** The output GGUF won't load.

**Troubleshooting:**
1. Load the original GGUF to confirm it's valid
2. Try a simple scalar patch first
3. Check disk space (output needs ~2× input size)
4. Verify write permissions on output directory

---

## Batch operations applied in wrong order

**Problem:** Operations were applied differently than expected.

**Note:** Operations are applied **in the order you added them**. If two operations touch the same tensor, the second modifies the result of the first.

**Example:**
- Op 1: Transform tensor X by scale=2
- Op 2: Transform tensor X by bias=+0.5
- Result: X is scaled first, then biased

---

## Tips

- Start with small scale/bias values (e.g., 0.95–1.05, ±0.01)
- Large changes often break the model
- Test incrementally - apply a batch, test the model, iterate
- Keep original GGUF files safe