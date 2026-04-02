# GGUF Workbench V2.1

A focused Python + Gradio tool for **surgically editing GGUF model weights**. Make targeted edits to float tensors, queue up multiple changes, and export a single modified model.

---

## 📋 Table of Contents

1. [What This Does](#what-this-does)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [UI Tabs & Workflows](#ui-tabs--workflows)
5. [Batch Operations (New!)](#batch-operations)
6. [Examples](#examples)
7. [Scope & Limitations](#scope--limitations)
8. [Troubleshooting](#troubleshooting)

---

## What This Does

GGUF Workbench lets you:

- **Inspect** GGUF model structure: metadata, tensors, data types, shapes
- **Edit individual weights** by index (scalar patching)
- **Transform tensor slices** using math formulas (`new = old * scale + bias`)
- **Patch entire tensors** along a single axis
- **Batch multiple edits** and apply them in one go to create a final modified model

**Perfect for:**
- Fine-tuning weight values post-training
- Experimenting with scaling factors on embeddings or attention heads
- Zeroing out specific neurons or rows
- A/B testing weight modifications
- Model surgery without retraining

---

## Installation

### Prerequisites
- Python 3.8+
- macOS, Linux, or WSL on Windows

### Setup

```bash
# Clone the repo or navigate to the directory
cd gguf_workbench_v2_1

# Create a virtual environment
python -m venv .venv

# Activate it
source .venv/bin/activate  # macOS/Linux
# OR
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Run

```bash
python app.py
```

Then open your browser to the local URL (usually `http://localhost:7860`).

---

## Quick Start

1. **Load a GGUF model**
   - Paste the full path to your `.gguf` file in the textbox at the top
   - Click **Load GGUF**
   - You'll see the model's metadata and tensor list

2. **Explore what you can edit**
   - Go to **Metadata + Tensors** tab
   - Click **"Editable only"** to filter for editable tensors
   - You can search by name (e.g., "attn", "mlp", "embed")

3. **Make a single edit**
   - Go to **Patch scalar**, **Transform**, or **Patch slice**
   - Select a tensor
   - Set your parameters
   - Click **Apply now** to save immediately

4. **Or queue multiple edits (Batch)**
   - Use **Add to batch** buttons instead of **Apply now**
   - Go to **Batch Manager** to see your queue
   - When done, click **Apply batch** to write one final model

---

## UI Tabs & Workflows

### **Metadata + Tensors**

View the structure of your loaded GGUF.

- **Metadata table**: All key-value pairs (model name, tokenizer info, etc.)
- **Tensor list**: All tensors with shape, type, element count, and editability status
- **Filter**: Search by tensor name (case-insensitive, substring match)
- **Editable only**: Only show float-like tensors (F32, F16, BF16)

**Use this to:**
- Understand your model's architecture
- Find the tensor name you want to edit
- Verify a tensor is editable before attempting to patch

---

### **Inspect**

Deep-dive into a single tensor's values.

1. Select a tensor from the dropdown
2. Choose how to decode it (usually "auto" is fine)
3. Set preview size (how many values to show)
4. Click **Inspect**

**Output:**
- Tensor shape, type, element count
- Min / Max / Mean / Std statistics
- Flattened preview table (first N values)

**Use this to:**
- See if a tensor's values are reasonable
- Decide what scaling factor to apply
- Debug strange model behavior

---

### **Patch scalar**

Edit exactly one number in a tensor.

1. Select the tensor
2. Enter indices as comma-separated coordinates (e.g., `0,1,2` for `tensor[0,1,2]`)
3. Enter the new value
4. **Either:**
   - Click **Apply now** to save immediately, OR
   - Click **Add to batch** to queue this for later

**Output:** A new GGUF with only that one weight changed.

**Example:**
- Tensor: `model.embed.weight` (shape `[32000, 4096]`)
- Indices: `100,50` (the embedding at index 100, dimension 50)
- New value: `0.5`
- Result: That one float is now `0.5`

---

### **Transform whole tensor**

Apply a formula to every value: `new = old * scale + bias`

1. Select the tensor
2. Set **Scale** (multiply all values by this)
3. Set **Bias** (add this to all values)
4. Optionally set **Clip min** / **Clip max** to cap values
5. Click **Preview transform** to see before/after stats
6. **Either:**
   - Click **Apply now**, OR
   - Click **Add to batch**

**Output:** A new GGUF with the entire tensor transformed.

**Example 1: Dampen attention**
- Tensor: `attention.weight`
- Scale: `0.9`
- Bias: `0.0`
- Effect: All values multiplied by 0.9 (10% dampening)

**Example 2: Add noise**
- Tensor: `embedding.weight`
- Scale: `1.0`
- Bias: `0.1`
- Effect: All values increased by 0.1

**Example 3: Scale and clip**
- Tensor: `logits.weight`
- Scale: `0.5`
- Bias: `0.0`
- Clip min/max: `-1.0` / `1.0`
- Effect: All values halved, clamped to [-1, 1]

---

### **Patch slice**

Edit one slice of a tensor along a specific axis.

1. Select the tensor
2. Choose **Axis** (0, 1, 2, etc.)
3. Choose **Index** along that axis
4. Choose **Mode**:
   - `set_constant`: Replace the entire slice with one value
   - `scale_and_bias`: Apply `slice = slice * scale + bias`
5. Set parameters
6. Click **Preview slice edit**, then **Apply now** or **Add to batch**

**Output:** A new GGUF with one slice modified.

**Example: Zero out a single attention head**
- Tensor: `model.layer.0.self_attn.v_proj` (shape `[4096, 4096]`)
- Axis: `0`
- Index: `128`
- Mode: `set_constant`
- Value: `0.0`
- Effect: Rows 0-4095 of row 128 are set to `0.0`

---

### **Batch Manager** ⭐ NEW

Queue multiple edits and apply them together. This is the key to complex model surgery workflows.

**Workflow:**
1. Use the **Patch scalar**, **Transform**, or **Patch slice** tabs
2. For each edit, click **Add to batch** instead of **Apply now**
3. Go to **Batch Manager** tab
4. Review your queued operations
5. Enter an output path
6. Click **Apply batch**

**How it works:**
- Each operation is queued with its parameters
- The original GGUF is copied
- Operations are applied in order
- Result is saved as a new GGUF

**Advantages:**
- Single output file instead of many intermediate files
- Cleaner workflow for complex edits
- Easy to review what, you'll do before committing
- Can clear the queue and start over if needed

**Example: Adjust a new LoRA layer**
1. Patch scalar: `lora.linear.weight[0,0]` = 0.05
2. Patch scalar: `lora.linear.weight[0,1]` = 0.03
3. Transform: `lora.bias` × 0.8 + 0
4. Patch slice: `lora.gate` axis 1, index 5, set to 1.0
5. Go to Batch Manager
6. Output: `/models/my-model.lora.gguf`
7. Click **Apply batch**
8. Result: Single file with all 4 operations

---

## Batch Operations

### When to Batch

**Use batch mode when you:**
- Need to make 2+ edits to the same model
- Want a single output file
- Like to review all changes before committing
- Are experimenting and want to keep originals clean

**Use immediate mode ("Apply now") when you:**
- Making a single quick edit
- Want instant feedback
- Don't mind intermediate files

### How to Clear or Undo

- **Before applying:** Go to **Batch Manager**, click **Clear batch**
- **After applying:** The original GGUF is unchanged. Just load it again.

---

## Examples

### Example 1: Reduce model sensitivity

**Goal:** Scale down the logits of a model to make output probabilities less sharp.

```
1. Load model: /models/llama2.gguf
2. Go to Transform tab
3. Tensor: "output.weight"
4. Scale: 0.7
5. Bias: 0.0
6. Click Add to batch
7. Go to Batch Manager
8. Output: /models/llama2-reduced.gguf
9. Click Apply batch
```

Result: All output logits are 30% smaller, leading to softer probabilities.

---

### Example 2: Zero out specific token embeddings

**Goal:** Disable predictions for certain token IDs (e.g., malicious tokens).

```
1. Load model: /models/gpt3.gguf
2. Go to Inspect
3. Tensor: "token_embd.weight" (usually shape [vocab_size, embedding_dim])
4. Inspect to see values
5. Go to Patch scalar
6. Tensor: "token_embd.weight"
7. For each token ID T you want to disable:
   - Indices: "T,0"
   - Value: 0.0
   - Add to batch
   (Repeat for "T,1", "T,2", etc., or use Patch slice)
8. Go to Batch Manager
9. Apply batch
```

Result: Selected tokens are zeroed out.

---

### Example 3: Amplify certain attention heads

**Goal:** Strengthen specific attention patterns (e.g., token position attention).

```
1. Load: /models/base.gguf
2. Go to Transform
3. Tensor: "attn.0.heads.3.weight"  (head 3 of layer 0)
4. Scale: 1.5
5. Bias: 0.0
6. Add to batch
7. Repeat for other heads you want to amplify
8. Batch Manager → Apply batch
```

Result: Selected attention heads have 50% stronger weights.

---

### Example 4: Fine-tune a layer post-training

**Goal:** Make small adjustments to an existing trained layer.

```
1. Load: /models/trained.gguf
2. Inspect: "mlp.layer1.weight" to see current values (e.g., mean=0.02, std=0.15)
3. Transform:
   Tensor: "mlp.layer1.weight"
   Scale: 1.1  (10% stronger)
   Bias: 0.0
   Add to batch
4. Transform:
   Tensor: "mlp.layer1.bias"
   Scale: 1.0
   Bias: 0.005  (slight positive shift)
   Add to batch
5. Batch Manager → Apply batch → /models/tuned.gguf
```

Result: Layer 1 weights are slightly tuned without retraining.

---

## Scope & Limitations

### ✅ Supported Tensor Types

- **F32** (32-bit float)
- **F16** (16-bit half precision)
- **BF16** (bfloat16)

### ❌ Not Supported (Intentionally)

- **Quantized types**: Q4_K, Q5_K, Q6_K, IQ*, etc.
- **Integer types**: i32, u8, etc.

**Why?** Quantized types require knowledge of scale factors and block structures. This tool focuses on the simpler case of float tensors.

### Other Limitations

- **Immutable metadata:** You can't change model name, architecture, etc. Only tensor weights.
- **No automatic validation:** The tool doesn't check if your edits break the model. You're responsible for sensible values.
- **Single-GPU friendly:** No distributed editing; it's all sequential and CPU-friendly.
- **File size unchanged:** Output file is the same size as the input (only tensor data changes).

---

## Troubleshooting

### "Tensor not found"

**Problem:** You can't find a tensor you're looking for.

**Solution:**
1. Go to **Metadata + Tensors**
2. Use the filter box to search
3. Note exact tensor names are case-sensitive
4. Check **Editable only** to see what's actually editable

---

### "Indices out of bounds"

**Problem:** You entered indices like `10,5,2` but the tensor is smaller.

**Solution:**
1. Go to **Inspect** and check the tensor shape
2. Verify your indices are within the bounds
3. Example: If shape is `[100, 50, 10]`, valid indices are `[0-99, 0-49, 0-9]`

---

### "Tensor is not editable"

**Problem:** You selected a quantized tensor (e.g., Q4_K).

**Solution:**
1. This workbench only edits float tensors by design
2. Look for other tensors with types F32, F16, or BF16
3. If you really need to edit quantized tensors, consider dequantizing the model first elsewhere

---

### Output file is corrupted

**Problem:** The output GGUF won't load.

**Troubleshooting:**
1. **Check the original:** Load the original GGUF to confirm it's valid
2. **Try a simple edit:** Test with a single scalar patch to rule out complex edge cases
3. **Check disk space:** Ensure you have enough space for the output file (usually 2× input size)
4. **File permissions:** Verify you can write to the output directory

---

### Batch operations applied in wrong order

**Problem:** You added operations to batch, but they were applied differently than expected.

**Note:** Operations are applied **in the order you added them**, to the same tensor copy. If two operations touch the same tensor, the second one will modify the result of the first.

**Example:**
- Op 1: Transform tensor X by scale=2
- Op 2: Transform tensor X by bias=+0.5
- Result: tensor X is scaled first, then biased (not the other way)

---

## Tips & Best Practices

1. **Always inspect first:** Before editing, use the **Inspect** tab to understand the tensor's current values.

2. **Use Preview:** For Transform and Slice operations, always click **Preview** before applying.

3. **Small changes:** Start with small scale/bias values (e.g., 0.95–1.05, ±0.01). Large changes often break the model.

4. **Backup originals:** Keep your original GGUF files. This tool doesn't include undo.

5. **Test incrementally:** Apply a batch, test the modifiedmodel on a simple prompt, then iterate.

6. **Use batch for related edits:** If you're modifying multiple parts of the same layer, use batch mode to keep things organized.

7. **Name your outputs clearly:** Use suffixes like `.reduced`, `.amplified`, `.tuned` to track what changes you made.

---

## Contributing

Found a bug or want a feature? Feel free to open an issue or PR.

---

## License

MIT (use freely, modify as needed)

---

**Happy editing! 🚀**
