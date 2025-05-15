import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from safetensors.torch import load_file, save_file

console = Console()


def convert_checkpoint(input_path: str, output_path: str, to_comfy: bool = True) -> None:
    """
    Convert checkpoint format between Diffusers and ComfyUI formats

    Args:
        input_path: Path to input safetensors file
        output_path: Path to save converted safetensors file
        to_comfy: If True, converts "transformer." to "diffusion_model."
                 If False, converts "diffusion_model." to "transformer."
    """
    # Load the state dict
    state_dict = load_file(input_path)

    # Convert the keys
    source_prefix = "transformer." if to_comfy else "diffusion_model."
    target_prefix = "diffusion_model." if to_comfy else "transformer."
    format_name = "ComfyUI" if to_comfy else "Diffusers"

    converted_state_dict = {}
    replaced_count = 0
    for k, v in state_dict.items():
        new_key = k.replace(source_prefix, target_prefix)
        converted_state_dict[new_key] = v
        if new_key != k:
            replaced_count += 1

    if replaced_count == 0:
        console.print(
            f"No keys were converted. The checkpoint may already be in {format_name} format or "
            f"doesn't contain '{source_prefix}' keys."
        )
        console.print("[red]Aborting[/red]")
        sys.exit(1)

    # Save the converted state dict
    save_file(converted_state_dict, output_path)
    console.print(f"Converted {replaced_count} keys from '{source_prefix}' to '{target_prefix}'")


app = typer.Typer(help="Convert checkpoint format between Diffusers and ComfyUI formats")


@app.command()
def main(
    input_path: str = typer.Argument(..., help="Path to input safetensors file"),
    to_comfy: bool = typer.Option(
        False, "--to-comfy", help="Convert from transformer to diffusion_model prefix (ComfyUI format)"
    ),
    output_path: Optional[str] = typer.Option(
        None,
        "--output-path",
        help="Path to save converted safetensors file. If not provided, will use input filename with suffix.",
    ),
) -> None:
    input_path = Path(input_path)
    if not input_path.exists():
        console.print(f"[bold red]Error:[/bold red] Input file not found: {input_path}")
        sys.exit(1)

    if output_path:
        output_path = Path(output_path)
    else:
        # Auto-generate output path by adding suffix to input filename
        suffix = "_comfy" if to_comfy else "_diffusers"
        # Remove existing _comfy or _diffusers suffix if present
        stem = input_path.stem
        if stem.endswith(("_comfy", "_diffusers")):
            stem = stem.rsplit("_", 1)[0]
        output_path = input_path.parent / f"{stem}{suffix}{input_path.suffix}"

    console.print(f"Converting {input_path} -> {output_path}")
    console.print(f"Direction: {'Diffusers -> ComfyUI' if to_comfy else 'ComfyUI -> Diffusers'}")

    convert_checkpoint(str(input_path), str(output_path), to_comfy)
    console.print("[bold green]Conversion complete![/bold green]")


if __name__ == "__main__":
    app()
