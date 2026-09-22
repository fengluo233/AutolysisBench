"""Restore a directory of autolyzed RGB patches using the released liver model."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from PIL import Image

from data.base_dataset import get_transform
from data.image_folder import IMG_EXTENSIONS
from models import create_model
from options.test_options import TestOptions
from runtime import require_cuda, seed_everything
from util.util import tensor2im


class Restorer:
    """Use the original SB forward path, including its random-number sequence.

    B/B2 are unused reference placeholders during restoration. Passing A in these
    slots avoids needing target images without changing the original forward code.
    """

    def __init__(self, checkpoint, device="cuda:0", seed=42):
        checkpoint = Path(checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError("Checkpoint not found: %s" % checkpoint)
        index = require_cuda(device)
        seed_everything(seed)
        self.opt = TestOptions(cmd_line="").gather_options()
        self.opt.isTrain = False
        self.opt.gpu_ids = [index]
        self.opt.no_flip = True
        self.opt.eval = True
        self.opt.batch_size = 1
        self.opt.serial_batches = True
        self.opt.num_threads = 0
        self.model = create_model(self.opt)
        self.transform = get_transform(self.opt)
        self.checkpoint = checkpoint
        self.initialized = False

    def restore(self, image):
        tensor = self.transform(image.convert("RGB")).unsqueeze(0)
        data = {"A": tensor, "B": tensor, "A_paths": ["input"], "B_paths": ["input"]}
        if not self.initialized:
            # Preserve the initialization forward pass used by the source code.
            self.model.data_dependent_initialize(data, data)
            state = torch.load(str(self.checkpoint), map_location="cpu")
            if hasattr(state, "_metadata"):
                del state._metadata
            self.model.netG.load_state_dict(state, strict=True)
            self.model.parallelize()
            self.model.eval()
            self.initialized = True
        self.model.set_input(data, data)
        self.model.test()
        outputs = {}
        for step in range(1, 6):
            value = getattr(self.model, "fake_%d" % step)
            if not torch.isfinite(value).all():
                raise RuntimeError("Non-finite output at step %d" % step)
            outputs[step] = Image.fromarray(tensor2im(value))
        return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_steps", action="store_true", help="Also save steps 1-4")
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error("Checkpoint not found: %s" % args.checkpoint)
    if not args.input.is_dir():
        parser.error("Input directory not found: %s" % args.input)
    extensions = {x.lower() for x in IMG_EXTENSIONS}
    paths = sorted(p for p in args.input.rglob("*") if p.is_file() and p.suffix.lower() in extensions)
    if not paths:
        parser.error("Input directory contains no supported images")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output directory must be new or empty; existing results are not overwritten")
    if args.output.resolve() == args.input.resolve() or args.input.resolve() in args.output.resolve().parents:
        parser.error("Output must be outside the input directory")
    # Validate before starting so a broken image is never silently skipped.
    for path in paths:
        try:
            with Image.open(path) as image:
                image.verify()
        except (OSError, ValueError) as error:
            parser.error("Cannot read image %s: %s" % (path, error))
    restorer = Restorer(args.checkpoint, args.device, args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for i, path in enumerate(paths, 1):
        with Image.open(path) as image:
            outputs = restorer.restore(image)
        relative = path.relative_to(args.input)
        # Retain the input extension in the name to avoid a.jpg/a.png collisions.
        for step in (range(1, 6) if args.save_steps else [5]):
            target = args.output / ("fake_%d" % step) / relative.parent / (relative.name + ".png")
            target.parent.mkdir(parents=True, exist_ok=True)
            outputs[step].save(target)
        records.append(relative.as_posix())
        print("[%d/%d] %s" % (i, len(paths), relative), flush=True)
    metadata = {"checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                "seed": args.seed, "steps": 5, "tau": 0.01,
                "preprocess": "RGB, bicubic resize 256, normalize [-1,1], no flip",
                "device": args.device, "torch": torch.__version__, "inputs": records}
    (args.output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
