# Ticket Template

## Metadata
- **ID**: `015`
- **Title**: Refactor app.py into modular structure + add repo graph for AI agents
- **Priority**: P1
- **Status**: done
- **Created**: 2026-04-05
- **Updated**: 2026-04-05

## Problem
The current `app.py` is a monolithic 1500+ line file containing:
- GGUF parsing logic (BinaryReader, GGUFParser, TensorInfo, GGUFManifest)
- Data transformation functions (decode/encode tensors, BF16 conversion)
- UI building functions (Gradio blocks construction)
- Event handlers (on_load, filter_tensor_table, inspect_tensor, etc.)
- Batch operation logic

This makes the codebase difficult to:
- Navigate for AI agents or new developers
- Test individual components in isolation
- Maintain and extend specific features
- Understand the overall architecture

Additionally, there's no structured documentation for AI agents to understand the codebase structure and dependencies.

## Proposed Solution

### Part 1: Modularize app.py into a package structure

Create a `gguf_workbench/` Python package with:

```
gguf_workbench/
├── __init__.py           # Main exports
├── parser.py             # GGUF parsing (BinaryReader, GGUFParser, TensorInfo, GGUFManifest)
├── tensor_ops.py         # Tensor decode/encode/transform operations
├── validation.py         # Output path validation, index parsing
├── compare.py            # GGUF comparison logic
├── batch.py              # Batch operation logic
├── app.py                # Gradio UI (only the build_app and event handlers)
└── constants.py          # All constants (GGUF_TYPE_*, GGML_TYPE_*, etc.)
```

**Migration rules:**
- Each module gets its own tests in `tests/test_<module>.py`
- Imports between modules use relative imports
- Functions remain backward compatible (keep wrappers if needed)
- All existing functionality preserved

### Part 2: Create repo graph/manifest for AI agents

Create `REPO_STRUCTURE.md` at root that documents:
- Directory structure and purpose of each folder
- Key modules and their responsibilities
- Entry points (how to run, test)
- Dependency graph
- Quick reference for common tasks

Also create a `.repo_manifest.json` with structured metadata for programmatic use.

## Acceptance Criteria
- [ ] app.py split into at least 5 logically distinct modules
- [ ] All existing tests pass after refactoring
- [ ] `python app.py` still works (no breaking changes)
- [ ] REPO_STRUCTURE.md exists and accurately describes the codebase
- [ ] .repo_manifest.json exists with valid JSON
- [ ] Gradio UI renders and functions identically to before

## Notes
- Use existing test structure as model (test_*.py pattern)
- Keep backwards compatibility - don't change public API signatures
- The refactoring is purely organizational, no logic changes

## Dependencies
None - this is a standalone refactoring task.