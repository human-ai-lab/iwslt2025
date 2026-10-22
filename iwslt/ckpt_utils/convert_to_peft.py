#!/usr/bin/env python3
"""Convert a full Fairseq checkpoint into a PEFT adapter checkpoint directory."""

from __future__ import annotations

import argparse
import copy
import os
import sys
from collections import OrderedDict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a full IWSLT Fairseq checkpoint into a compact checkpoint "
            "plus a Hugging Face PEFT adapter."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Full Fairseq checkpoint, for example avg_best10.pt.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output PEFT checkpoint directory, for example best_ckpt.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        help=(
            "Optional Fairseq data directory. Use this when the data path stored "
            "in the checkpoint is unavailable on the current machine."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow writing into an existing non-empty output directory.",
    )
    return parser.parse_args()


def should_extract_parameter(name: str) -> bool:
    """Return whether a parameter is saved separately or supplied by a base model."""
    belongs_to_base_model = "llama" in name or "whisper_encoder" in name
    belongs_to_projector = "llama_proj" in name or "avgpool" in name
    return belongs_to_base_model and not belongs_to_projector


def prioritize_bundled_nvjitlink() -> None:
    """Restart with the virtualenv nvJitLink ahead of incompatible system CUDA."""
    python_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    nvjitlink_dir = (
        Path(sys.prefix)
        / "lib"
        / python_version
        / "site-packages"
        / "nvidia"
        / "nvjitlink"
        / "lib"
    )
    if not nvjitlink_dir.is_dir():
        return

    current_dirs = [
        entry for entry in os.environ.get("LD_LIBRARY_PATH", "").split(":") if entry
    ]
    nvjitlink_dir_str = str(nvjitlink_dir)
    if current_dirs and current_dirs[0] == nvjitlink_dir_str:
        return

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(
        [nvjitlink_dir_str]
        + [entry for entry in current_dirs if entry != nvjitlink_dir_str]
    )
    script_path = str(Path(__file__).resolve())
    os.execve(
        sys.executable,
        [sys.executable, script_path, *sys.argv[1:]],
        env,
    )


def convert(
    input_path: Path,
    output_path: Path,
    data_root: Path | None = None,
    force: bool = False,
) -> None:
    if not input_path.is_file():
        raise FileNotFoundError(f"Input checkpoint not found: {input_path}")
    if output_path.exists() and not output_path.is_dir():
        raise NotADirectoryError(f"Output path is not a directory: {output_path}")
    if output_path.is_dir() and any(output_path.iterdir()) and not force:
        raise FileExistsError(
            f"Output directory is not empty: {output_path}. "
            "Pass --force to overwrite generated files."
        )

    prioritize_bundled_nvjitlink()

    # Import project and Fairseq modules only after argument validation so that
    # `--help` does not initialize the model stack.
    project_root = str(Path(__file__).resolve().parents[2])
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    import torch
    from fairseq import checkpoint_utils, tasks, utils

    import iwslt  # noqa: F401 - registers custom tasks
    import iwslt.models  # noqa: F401 - registers custom models

    print(f"Loading checkpoint: {input_path}")
    state = checkpoint_utils.load_checkpoint_to_cpu(str(input_path))
    if "cfg" not in state or "task" not in state["cfg"] or "model" not in state["cfg"]:
        raise KeyError("Checkpoint does not contain cfg.task and cfg.model.")
    if "model" not in state:
        raise KeyError("Checkpoint does not contain model parameters.")

    task_cfg = copy.deepcopy(state["cfg"]["task"])
    if data_root is not None:
        task_cfg.data = str(data_root)

    task = tasks.setup_task(task_cfg)
    model = task.build_model(state["cfg"]["model"])
    load_result = model.load_state_dict(state["model"], strict=True)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError(
            "Checkpoint did not load cleanly: "
            f"missing={load_result.missing_keys}, "
            f"unexpected={load_result.unexpected_keys}"
        )
    if not hasattr(model.llama, "peft_config"):
        raise TypeError("The checkpoint model is not configured as a PEFT model.")

    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Saving PEFT adapter: {output_path}")
    model.llama.save_pretrained(
        str(output_path),
        save_embedding_layers=True,
        safe_serialization=True,
    )

    reduced_state = dict(state)
    reduced_state["model"] = OrderedDict(
        (name, value)
        for name, value in state["model"].items()
        if not should_extract_parameter(name)
    )
    reduced_state = utils.move_to_cpu(reduced_state)
    compact_checkpoint = output_path / "checkpoint.pt"
    torch.save(reduced_state, compact_checkpoint)

    removed_count = len(state["model"]) - len(reduced_state["model"])
    print(
        f"Finished: {compact_checkpoint} "
        f"({removed_count} base-model parameters extracted)."
    )


def main() -> None:
    args = parse_args()
    convert(
        input_path=args.input,
        output_path=args.output,
        data_root=args.data_root,
        force=args.force,
    )


if __name__ == "__main__":
    main()
