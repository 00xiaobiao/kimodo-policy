import argparse, json, tempfile, random
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset

def stat(x, name):
    a = x.abs()
    print(f"{name}: shape={tuple(x.shape)} min={x.min():.4f} max={x.max():.4f}\n"
          f"mean_abs={a.mean():.4f} max_abs={a.max():.4f} "
          f"out1={(a>1).float().mean():.4f} out2={(a>2).float().mean():.4f} out3={(a>3).float().mean():.4f}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--tokenizer", default="")
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    ckpt = Path(args.ckpt)
    tmp = None
    if args.tokenizer:
        tmp = tempfile.TemporaryDirectory(prefix="pi05_compat_")
        compat = Path(tmp.name)
        for child in ckpt.iterdir():
            (compat / child.name).symlink_to(child, target_is_directory=child.is_dir())
        pp = json.loads((ckpt / "policy_preprocessor.json").read_text())
        for step in pp["steps"]:
            if step.get("registry_name") == "tokenizer_processor":
                step["config"]["tokenizer_name"] = args.tokenizer
        (compat / "policy_preprocessor.json").unlink()
        (compat / "policy_preprocessor.json").write_text(json.dumps(pp, indent=2) + "\n")
        ckpt = compat

    config = PreTrainedConfig.from_pretrained(ckpt)
    config.device = args.device
    policy = get_policy_class(config.type).from_pretrained(ckpt, config=config).to(args.device).eval()

    preprocessor, _ = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=ckpt,
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )

    meta = LeRobotDatasetMetadata(args.dataset)
    delta = resolve_delta_timestamps(config, meta)
    dataset = LeRobotDataset(args.dataset, delta_timestamps=delta)

    preds, gts = [], []
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= args.n: break
            torch.manual_seed(args.seed + i)
            torch.cuda.manual_seed_all(args.seed + i)

            processed = preprocessor(batch)
            noise = torch.randn(
                1, config.chunk_size, config.max_action_dim,
                device=args.device,
            )
            pred = policy.predict_action_chunk(processed, noise=noise)
            preds.append(pred.squeeze(0).cpu())
            gts.append(processed["action"].squeeze(0).cpu())

    pred = torch.stack(preds)
    gt = torch.stack(gts)

    print("GLOBAL")
    stat(pred.reshape(-1, 40), "pred_norm")
    stat(gt.reshape(-1, 40), "gt_norm")

    print("FIRST")
    stat(pred[:, 0, :], "pred_first")
    stat(gt[:, 0, :], "gt_first")

    print("DIFF")
    stat((pred - gt), "pred_minus_gt")

    if tmp: tmp.cleanup()

if __name__ == "__main__":
    main()
