#!/usr/bin/env python
"""Monte Carlo runner for simulation baseline neural-network methods."""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam

from simulation_common import (
    DataGenerator,
    FeedForward,
    SplitData,
    baseline_label,
    load_feedforward_model,
    make_baseline_features,
    default_response_error_sd,
    paper_measurement_error_sd,
    parse_int_list,
    resolve_device,
    save_feedforward_model,
    set_seed,
    write_or_replace_row,
)


CSV_FIELDS = [
    "seed",
    "case",
    "method",
    "label",
    "best_epoch",
    "best_valid_mse",
    "test_mse",
    "elapsed_seconds",
    "n_samples",
    "n_grid",
    "me",
    "err",
    "batch_size",
    "epochs",
    "lr",
    "hidden",
    "dropout",
    "split_seed",
    "device",
    "input_dim",
    "n_basis",
    "target_fve",
    "achieved_fve",
]


def train_one(args: argparse.Namespace) -> dict:
    set_seed(args.seed)
    device = resolve_device(args.device)
    measurement_error = args.me
    response_error = args.err
    if measurement_error is None:
        measurement_error = paper_measurement_error_sd(args.case)
    if response_error is None:
        response_error = default_response_error_sd(args.case)

    grid = np.linspace(0.0, 1.0, args.n_grid)
    generator = DataGenerator(grid, case=args.case, me=measurement_error, err=response_error)
    x, y, t = generator.generate(args.n_samples)
    features, feature_info = make_baseline_features(
        args.method,
        x,
        t,
        split=tuple(args.split),
        split_seed=args.split_seed,
        n_basis=args.n_basis,
        fve=args.fve,
    )

    data = SplitData(
        features,
        y,
        t,
        batch_size=args.batch_size,
        split=tuple(args.split),
        seed=args.split_seed,
    )

    hidden = parse_int_list(args.hidden)
    model = FeedForward(
        in_d=features.shape[1],
        hidden=hidden,
        dropout=args.dropout,
    ).to(device)

    epoch = args.epochs
    pred_loss_train_history = []
    loss_valid_history = []
    optimizer = Adam(model.parameters(), lr=args.lr)
    compute_loss = torch.nn.MSELoss()
    min_valid_loss = sys.maxsize
    start_time = time.time()
    label = baseline_label(args.method, n_basis=args.n_basis, fve=args.fve)

    with tempfile.TemporaryDirectory(prefix=f"baseline_{label}_seed_{args.seed}_") as tmp_dir:
        folder = tmp_dir + "/"
        Path(folder).mkdir(parents=True, exist_ok=True)
        best_epoch = 0

        for k in range(epoch):
            pred_loss_train = []
            loss_valid = []
            data.shuffle()
            model.train()

            for i, (x, y) in enumerate(data.get_train_batch()):
                x, y = x.to(device), y.to(device)
                out = model.forward(x)
                loss_pred = compute_loss(out, y)
                loss = loss_pred
                pred_loss_train.append(loss_pred.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            pred_loss_train_history.append(np.mean(pred_loss_train))

            with torch.no_grad():
                model.eval()
                for x, y in data.get_valid_batch():
                    x, y = x.to(device), y.to(device)
                    valid_y = model.forward(x)
                    valid_loss = compute_loss(valid_y, y)
                    loss_valid.append(valid_loss.item())

            if np.mean(loss_valid) < min_valid_loss:
                save_feedforward_model(folder, "best", features.shape[1], hidden, args.dropout, model, optimizer)
                min_valid_loss = np.mean(loss_valid)
                best_epoch = k + 1

            loss_valid_history.append(np.mean(loss_valid))

            if (k+1) % 50 == 0:
                print("epoch:", k+1, "\n",
                      "prediction training loss = ", pred_loss_train_history[-1],
                      "validation loss = ", loss_valid_history[-1])

        ck = folder + "best_checkpoint.pth"
        model = load_feedforward_model(ck, device)

        loss_test = []

        with torch.no_grad():
            model.eval()
            for x, y in data.get_test_batch():
                x, y = x.to(device), y.to(device)
                test_y = model.forward(x)
                test_loss = compute_loss(test_y, y)
                loss_test.append(test_loss.item())

        test_mse = np.mean(loss_test)

    row = {
        "seed": args.seed,
        "case": args.case,
        "method": args.method,
        "label": label,
        "best_epoch": best_epoch,
        "best_valid_mse": min_valid_loss,
        "test_mse": test_mse,
        "elapsed_seconds": time.time() - start_time,
        "n_samples": args.n_samples,
        "n_grid": args.n_grid,
        "me": measurement_error,
        "err": response_error,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "hidden": ",".join(map(str, hidden)),
        "dropout": args.dropout,
        "split_seed": args.split_seed,
        "device": str(device),
        "input_dim": feature_info.get("input_dim", ""),
        "n_basis": feature_info.get("n_basis", ""),
        "target_fve": feature_info.get("target_fve", ""),
        "achieved_fve": feature_info.get("achieved_fve", ""),
    }

    output_root = args.output_root or f"results/case{args.case}_baselines_mc"
    csv_path = Path(output_root) / label / "mse.csv"
    write_or_replace_row(csv_path, row, CSV_FIELDS)

    print(
        f"wrote seed {args.seed} to {csv_path}; "
        f"test_mse={test_mse:.8g}, best_valid_mse={min_valid_loss:.8g}",
        flush=True,
    )
    return row


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["raw", "bspline", "fpca"], required=True)
    parser.add_argument("--case", type=int, choices=[1, 2, 3, 4], default=1)
    parser.add_argument("--output-root", default="")
    parser.add_argument("--device", default="cpu")

    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--n-samples", type=int, default=4000)
    parser.add_argument("--n-grid", type=int, default=51)
    parser.add_argument("--me", type=float, default=None)
    parser.add_argument("--err", type=float, default=None)
    parser.add_argument("--split", type=int, nargs=3, default=[64, 16, 20])

    parser.add_argument("--n-basis", type=int, default=4)
    parser.add_argument("--fve", type=float, default=0.9)

    parser.add_argument("--hidden", default="128,128,128")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.split_seed is None:
        args.split_seed = args.seed
    train_one(args)


if __name__ == "__main__":
    main()
