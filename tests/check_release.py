"""CUDA integration checks; only generated synthetic images are used.

python tests/check_release.py --checkpoint /path/to/released.pth --work-dir /tmp/check
An optional --reference-root compares against the original research checkout.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def run(args, expected=0):
    result = subprocess.run([sys.executable] + list(map(str, args)), cwd=str(ROOT),
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(result.stdout, flush=True)
    if (expected == 0 and result.returncode != 0) or (expected != 0 and result.returncode == 0):
        raise AssertionError("Unexpected exit code: %s" % result.returncode)
    return result.stdout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--reference-root", type=Path)
    args = parser.parse_args()
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=False)
    images = work / "input"
    images.mkdir()
    rng = np.random.RandomState(17)
    for i in range(2):
        pixels = rng.randint(0, 256, (300 + i * 20, 340, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(images / ("%d.png" % i))
    checkpoint = args.checkpoint.resolve()
    cmd = ["infer.py", "--input", images, "--checkpoint", checkpoint, "--seed", "42", "--save_steps"]
    run(cmd + ["--output", work / "out1"])
    run(cmd + ["--output", work / "out2"])
    for step in range(1, 6):
        for i in range(2):
            a = work / "out1" / ("fake_%d" % step) / ("%d.png.png" % i)
            b = work / "out2" / ("fake_%d" % step) / ("%d.png.png" % i)
            assert Image.open(a).size == (256, 256)
            assert a.read_bytes() == b.read_bytes(), "Seeded inference differs"
    run(cmd + ["--output", work / "out1"], expected=1)
    (work / "empty").mkdir()
    run(["infer.py", "--input", work / "empty", "--checkpoint", checkpoint,
         "--output", work / "unused"], expected=1)
    run(["infer.py", "--input", images, "--checkpoint", work / "missing.pth",
         "--output", work / "unused"], expected=1)
    (work / "bad").mkdir()
    (work / "bad/broken.png").write_bytes(b"not an image")
    run(["infer.py", "--input", work / "bad", "--checkpoint", checkpoint,
         "--output", work / "unused"], expected=1)

    data = work / "data"
    for domain in ["trainA", "trainB"]:
        (data / domain).mkdir(parents=True)
        for i in range(2):
            (data / domain / ("%d.png" % i)).symlink_to(images / ("%d.png" % i))
    run(["train.py", "--dataroot", data, "--checkpoints_dir", work / "checkpoints",
         "--name", "smoke", "--batch_size", "1", "--num_threads", "0",
         "--max_steps", "2", "--seed", "42"])
    run_dir = work / "checkpoints/smoke"
    losses = [json.loads(x) for x in (run_dir / "losses.jsonl").read_text().splitlines()]
    assert len(losses) == 2
    assert all(np.isfinite(v) for row in losses for v in row["losses"].values())
    # Recreate all four networks, initialize feature MLPs, and load every tensor.
    from options.train_options import TrainOptions
    from models import create_model
    from data import create_dataset
    from runtime import seed_everything
    opt = TrainOptions(cmd_line="--batch_size 1 --num_threads 0 --lambda_SB 1").gather_options()
    opt.isTrain = True
    opt.gpu_ids = [0]
    opt.dataroot = str(data)
    opt.continue_train = False
    opt.checkpoints_dir = str(work / "checkpoints")
    opt.name = "smoke"
    seed_everything(42)
    first, second = create_dataset(opt), create_dataset(opt)
    model = create_model(opt)
    initial = {k: v.detach().cpu().clone() for k, v in model.netG.state_dict().items()}
    model.data_dependent_initialize(next(iter(first)), next(iter(second)))
    for name in model.model_names:
        state = torch.load(str(run_dir / ("latest_net_%s.pth" % name)), map_location="cpu")
        assert all(torch.isfinite(v).all() for v in state.values())
        getattr(model, "net" + name).load_state_dict(state, strict=True)
    trained = model.netG.state_dict()
    change = max((trained[k].cpu() - v).abs().max().item() for k, v in initial.items())
    assert change > 0, "Generator did not update"
    del model
    torch.cuda.empty_cache()
    run(["infer.py", "--input", images, "--checkpoint", run_dir / "latest_net_G.pth",
         "--output", work / "trained_inference"])

    parity = None
    if args.reference_root:
        # Run the research model in a fresh interpreter; no imports are shared.
        script = '''
import sys, random
from pathlib import Path
import numpy as np
import torch
from PIL import Image
reference, checkpoint, images, output = sys.argv[1:]
sys.path.insert(0, reference)
sys.argv = ["reference_check"]
from options.test_options import TestOptions
from models import create_model
from data.base_dataset import get_transform
from util.util import tensor2im
random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
torch.cuda.set_device(0)
opt = TestOptions(cmd_line="").gather_options()
opt.isTrain=False; opt.gpu_ids=[0]; opt.no_flip=True; opt.eval=True
model=create_model(opt); transform=get_transform(opt)
torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
for i, p in enumerate(sorted(Path(images).glob("*.png"))):
    with Image.open(p) as im: x=transform(im.convert("RGB")).unsqueeze(0)
    data={"A":x,"B":x,"A_paths":[str(p)],"B_paths":[str(p)]}
    if i==0:
        model.data_dependent_initialize(data,data)
        state=torch.load(checkpoint,map_location="cpu")
        if hasattr(state,"_metadata"): del state._metadata
        model.netG.load_state_dict(state,strict=True)
        model.parallelize(); model.eval()
    model.set_input(data,data); model.test()
    for step in range(1,6):
        dest=Path(output)/("fake_%d"%step)/(p.name+".png")
        dest.parent.mkdir(parents=True,exist_ok=True)
        Image.fromarray(tensor2im(getattr(model,"fake_%d"%step))).save(dest)
'''
        run(["-c", script, args.reference_root, checkpoint, images, work / "reference"])
        parity = all(p.read_bytes() == (work / "reference" / p.relative_to(work / "out1")).read_bytes()
                     for p in (work / "out1").rglob("*.png"))
        assert parity, "Reference/release output differs"
    report = {"seeded_outputs_identical": True, "images": 2, "steps_checked": 5,
              "training_updates": 2, "finite_losses": True, "all_networks_reload": True,
              "generator_max_parameter_change": change, "trained_generator_inference": True,
              "invalid_input_checks": True, "reference_pixel_identical": parity,
              "torch": torch.__version__}
    (work / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
