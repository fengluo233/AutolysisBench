"""Train the liver A-to-F model with the original Schrodinger bridge optimization loop."""
import json
import math
import sys
from pathlib import Path

from data import create_dataset
from models import create_model
from options.train_options import TrainOptions
from runtime import require_cuda, seed_everything


def main():
    if not any(x == "--config" or x.startswith("--config=") for x in sys.argv[1:]):
        sys.argv.extend(["--config", str(Path(__file__).parent / "configs/liver_a2f.json")])
    options = TrainOptions()
    opt = options.gather_options()
    ids = [int(x) for x in opt.gpu_ids.split(",") if int(x) >= 0]
    if len(ids) != 1:
        raise ValueError("This release supports single-GPU training; pass --gpu_ids N")
    require_cuda("cuda:%d" % ids[0])
    if opt.batch_size < 1 or opt.max_steps < 0 or opt.n_epochs + opt.n_epochs_decay < opt.epoch_count:
        raise ValueError("Invalid batch size, max_steps or epoch range")
    opt.isTrain = True
    if opt.suffix:
        opt.name += "_" + opt.suffix.format(**vars(opt))
    run_dir = Path(opt.checkpoints_dir) / opt.name
    if run_dir.exists() and any(run_dir.iterdir()) and not opt.continue_train:
        raise ValueError("Training output exists; choose a new --name or use --continue_train")
    opt.gpu_ids = ids
    seed_everything(opt.seed)
    dataset = create_dataset(opt)
    dataset2 = create_dataset(opt)
    if dataset.dataset.A_size == 0 or dataset.dataset.B_size == 0:
        raise ValueError("Both trainA and trainB must contain images")
    if len(dataset) < opt.batch_size:
        raise ValueError("Dataset is smaller than batch_size; lower --batch_size")
    options.print_options(opt)
    model = create_model(opt)
    steps = 0
    total_images = 0
    with (run_dir / "losses.jsonl").open("a") as log:
        for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
            # Preserve the source loop's augmentation schedule for the two streams.
            dataset.set_epoch(epoch)
            for data, data2 in zip(dataset, dataset2):
                if steps == 0:
                    model.data_dependent_initialize(data, data2)
                    model.setup(opt)
                    model.parallelize()
                model.set_input(data, data2)
                model.optimize_parameters()
                steps += 1
                total_images += data["A"].size(0)
                losses = model.get_current_losses()
                if not all(math.isfinite(x) for x in losses.values()):
                    raise RuntimeError("Non-finite training loss at step %d" % steps)
                row = {"epoch": epoch, "step": steps, "images": total_images, "losses": losses}
                log.write(json.dumps(row) + "\n")
                log.flush()
                if steps == 1 or total_images % opt.print_freq == 0:
                    print(json.dumps(row), flush=True)
                if total_images % opt.save_latest_freq == 0:
                    model.save_networks("latest")
                if opt.max_steps and steps >= opt.max_steps:
                    model.save_networks("latest")
                    print("Smoke run finished after %d updates" % steps, flush=True)
                    return
            if epoch % opt.save_epoch_freq == 0 or epoch == opt.n_epochs + opt.n_epochs_decay:
                model.save_networks("latest")
                model.save_networks(epoch)
            model.update_learning_rate()
            print("End of epoch %d" % epoch, flush=True)


if __name__ == "__main__":
    main()
