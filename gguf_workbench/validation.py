import os
from pathlib import Path
from typing import Tuple


def validate_output_path(
    input_path: str, output_path: str, overwrite_confirm: bool = False
) -> Tuple[str, str]:
    """Validate output path before writing a patched GGUF file.

    Returns (resolved_output_path, warning_message).
    Raises ValueError on validation failure.
    """
    if not output_path or not output_path.strip():
        raise ValueError("Please specify an output path.")

    input_resolved = Path(input_path).resolve()
    output_resolved = Path(output_path.strip()).resolve()

    if input_resolved == output_resolved:
        raise ValueError(
            "Output path cannot be the same as the input file. Choose a different output path."
        )

    parent = output_resolved.parent
    if not parent.exists():
        try:
            parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise ValueError(f"Cannot create parent directory '{parent}': {e}")

    if not os.access(str(parent), os.W_OK):
        raise ValueError(f"Output directory is not writable: '{parent}'")

    warning = ""
    if output_resolved.exists():
        if not overwrite_confirm:
            raise ValueError(
                f"Output file already exists: '{output_path}'. "
                "Enable 'Confirm overwrite' to proceed, or choose a different path."
            )
        warning = f"Overwriting existing file: `{output_path}`\n\n"

    return str(output_resolved), warning


def default_output_path(input_path: str, suffix: str = ".patched.gguf") -> str:
    p = Path(input_path)
    return str(p.with_name(f"{p.stem}{suffix}"))
