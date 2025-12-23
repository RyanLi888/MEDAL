import argparse
import itertools
import json
import os
import subprocess
import sys
import time


def _run_trial(cmd, cwd):
    p = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return p.returncode, p.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_acc", type=float, default=0.95)
    ap.add_argument("--noise_rate", type=float, default=0.30)
    ap.add_argument("--data_split", type=str, default="train", choices=["train", "test"])
    ap.add_argument("--hc_impl", type=str, default="experimental", choices=["experimental", "original"])
    ap.add_argument("--max_trials", type=int, default=200)
    ap.add_argument("--output_root", type=str, default="/tmp/medal_auto_tune_runs")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    script_path = os.path.join(project_root, "experiments", "ml_vote_label_correction", "run_ml_vote_label_correction.py")

    try:
        os.makedirs(args.output_root, exist_ok=True)
    except OSError as e:
        raise SystemExit(
            f"❌ failed to create output_root={args.output_root} ({e}). "
            f"Hint: check disk space; on this machine /home is 100% full. "
            f"Try --output_root /tmp/medal_auto_tune_runs"
        )

    base = [
        sys.executable,
        script_path,
        "--mode",
        "from_preprocessed",
        "--data_split",
        args.data_split,
        "--noise_rate",
        str(args.noise_rate),
        "--seed",
        str(args.seed),
        "--hc_impl",
        args.hc_impl,
        "--skip_plots",
    ]

    # 优先搜索 keep_flip（用户确认允许 keep+flip 训练），其次 keep，最后 nondrop
    train_ons = ["keep_flip", "keep", "nondrop"]

    count_band_lows = [0, 1, 2, 3]
    count_band_highs = [5, 6, 7]

    prob_lows = [0.20, 0.25, 0.30, 0.40, 0.50]
    prob_highs = [0.54, 0.60, 0.70, 0.75, 0.80]

    mlp_hiddens = [128, 256]
    mlp_epochs = [200, 400]

    flip_train_weights = [0.3, 0.5, 0.7, 1.0]

    candidates = []

    for train_on in train_ons:
        weight_grid = flip_train_weights if train_on == "keep_flip" else [None]
        for ftw in weight_grid:
            for low, high in itertools.product(count_band_lows, count_band_highs):
                if low >= high:
                    continue
                candidates.append(
                    {
                        "train_on": train_on,
                        "flip_train_weight": ftw,
                        "vote_rule": "band",
                        "vote_score": "count",
                        "band_low": low,
                        "band_high": high,
                        "vote_k": 4,
                    }
                )

    for train_on in train_ons:
        weight_grid = flip_train_weights if train_on == "keep_flip" else [None]
        for ftw in weight_grid:
            for low, high in itertools.product(prob_lows, prob_highs):
                if low >= high:
                    continue
                candidates.append(
                    {
                        "train_on": train_on,
                        "flip_train_weight": ftw,
                        "vote_rule": "band",
                        "vote_score": "mean_proba",
                        "prob_low": low,
                        "prob_high": high,
                    }
                )

    for train_on in train_ons:
        weight_grid = flip_train_weights if train_on == "keep_flip" else [None]
        for ftw in weight_grid:
            for low, high in itertools.product(prob_lows, prob_highs):
                if low >= high:
                    continue
                candidates.append(
                    {
                        "train_on": train_on,
                        "flip_train_weight": ftw,
                        "vote_rule": "band",
                        "vote_score": "stacked",
                        "prob_low": low,
                        "prob_high": high,
                    }
                )

    for train_on in train_ons:
        weight_grid = flip_train_weights if train_on == "keep_flip" else [None]
        for ftw in weight_grid:
            for hidden, epochs, low, high in itertools.product(mlp_hiddens, mlp_epochs, prob_lows, prob_highs):
                if low >= high:
                    continue
                candidates.append(
                    {
                        "train_on": train_on,
                        "flip_train_weight": ftw,
                        "vote_rule": "band",
                        "vote_score": "mlp",
                        "mlp_hidden": hidden,
                        "mlp_epochs": epochs,
                        "mlp_lr": 1e-3,
                        "mlp_weight_decay": 1e-4,
                        "prob_low": low,
                        "prob_high": high,
                    }
                )

    ts = time.strftime("%Y%m%d_%H%M%S")
    best = {"acc": -1.0, "config": None, "trial_dir": None}

    results_path = os.path.join(args.output_root, f"results_{ts}.json")

    cache_dir = os.path.join(args.output_root, f"cache_{ts}")
    os.makedirs(cache_dir, exist_ok=True)
    cache_cmd = list(base)
    cache_cmd += [
        "--output_dir",
        cache_dir,
        "--train_on",
        "keep",
        "--vote_rule",
        "none",
        "--vote_score",
        "count",
        "--save_cache_dir",
        cache_dir,
    ]
    print(f"[cache] building cache at {cache_dir}")
    rc, out = _run_trial(cache_cmd, cwd=project_root)
    if rc != 0:
        with open(os.path.join(cache_dir, "cache_build.log"), "w", encoding="utf-8") as f:
            f.write(out)
        raise SystemExit(f"❌ cache build failed, see {os.path.join(cache_dir, 'cache_build.log')}")
    if not os.path.exists(os.path.join(cache_dir, "features.npy")):
        raise SystemExit(f"❌ cache build did not produce features.npy under {cache_dir}")

    tried = 0
    for i, cfg in enumerate(candidates):
        if tried >= args.max_trials:
            break
        trial_dir = os.path.join(args.output_root, f"trial_{ts}_{tried:04d}")
        os.makedirs(trial_dir, exist_ok=True)

        cmd = list(base)
        cmd += ["--reuse_cache_dir", cache_dir]
        cmd += ["--output_dir", trial_dir]

        cmd += ["--train_on", cfg["train_on"]]
        if cfg.get("flip_train_weight") is not None:
            cmd += ["--flip_train_weight", str(cfg["flip_train_weight"])]
        cmd += ["--vote_rule", cfg["vote_rule"]]
        cmd += ["--vote_score", cfg["vote_score"]]

        if cfg.get("vote_k") is not None:
            cmd += ["--vote_k", str(cfg["vote_k"])]
        if cfg.get("band_low") is not None:
            cmd += ["--band_low", str(cfg["band_low"])]
        if cfg.get("band_high") is not None:
            cmd += ["--band_high", str(cfg["band_high"])]
        if cfg.get("prob_low") is not None:
            cmd += ["--prob_low", str(cfg["prob_low"])]
        if cfg.get("prob_high") is not None:
            cmd += ["--prob_high", str(cfg["prob_high"])]
        if cfg.get("mlp_hidden") is not None:
            cmd += ["--mlp_hidden", str(cfg["mlp_hidden"])]
        if cfg.get("mlp_epochs") is not None:
            cmd += ["--mlp_epochs", str(cfg["mlp_epochs"])]
        if cfg.get("mlp_lr") is not None:
            cmd += ["--mlp_lr", str(cfg["mlp_lr"])]
        if cfg.get("mlp_weight_decay") is not None:
            cmd += ["--mlp_weight_decay", str(cfg["mlp_weight_decay"])]

        rc, out = _run_trial(cmd, cwd=project_root)

        summary_fp = os.path.join(trial_dir, "summary.json")
        if rc != 0 or (not os.path.exists(summary_fp)):
            with open(os.path.join(trial_dir, "run.log"), "w", encoding="utf-8") as f:
                f.write(out)
            tried += 1
            continue

        with open(summary_fp, "r", encoding="utf-8") as f:
            summary = json.load(f)
        acc = float(summary.get("correction_accuracy", summary.get("eval_all", {}).get("accuracy", 0.0)))

        record = {
            "trial": tried,
            "acc": acc,
            "config": cfg,
            "trial_dir": trial_dir,
        }

        if acc > best["acc"]:
            best = {"acc": acc, "config": cfg, "trial_dir": trial_dir}

        with open(results_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"trial={tried:04d} acc={acc:.4f} vote_score={cfg['vote_score']} train_on={cfg['train_on']} dir={trial_dir}")

        if acc >= args.target_acc:
            break

        tried += 1

    best_path = os.path.join(args.output_root, f"best_{ts}.json")
    with open(best_path, "w", encoding="utf-8") as f:
        json.dump(best, f, ensure_ascii=False, indent=2)

    print("\nBEST")
    print(json.dumps(best, ensure_ascii=False, indent=2))
    print(f"best_config_saved={best_path}")


if __name__ == "__main__":
    main()
