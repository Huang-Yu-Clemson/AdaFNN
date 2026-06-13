#!/usr/bin/env python
"""Monte Carlo runner for AdaFNN simulation cases."""

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
    AdaFNN,
    DataGenerator,
    SplitData,
    load_adafnn_model,
    parse_int_list,
    resolve_device,
    save_adafnn_model,
    set_seed,
    default_response_error_sd,
    paper_measurement_error_sd,
    tag_float,
    write_or_replace_row,
)


CSV_FIELDS = [
    "seed",
    "case",
    "lambda1",
    "lambda2",
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
    "n_base",
    "base_hidden",
    "sub_hidden",
    "dropout",
    "orth_pairs",
    "sparse_bases",
    "split_seed",
    "device",
]


def default_n_base(case: int) -> int:
    if case in (2, 3):
        return 3
    return 2


def train_one(args: argparse.Namespace) -> dict:
    set_seed(args.seed)
    device = resolve_device(args.device)
    measurement_error = args.me
    response_error = args.err
    if measurement_error is None:
        measurement_error = paper_measurement_error_sd(args.case)
    if response_error is None:
        response_error = default_response_error_sd(args.case)

    grid = np.linspace(0.0, 1.0, args.n_grid).tolist()
    generator = DataGenerator(grid, case=args.case, me=measurement_error, err=response_error)
    x, y, t = generator.generate(args.n_samples)

    data = SplitData(
        x,
        y,
        t,
        batch_size=args.batch_size,
        split=tuple(args.split),
        seed=args.split_seed,
    )

    base_hidden = parse_int_list(args.base_hidden)
    sub_hidden = parse_int_list(args.sub_hidden)
    n_base = args.n_base if args.n_base > 0 else default_n_base(args.case)
    model = AdaFNN(
        n_base=n_base,
        base_hidden=base_hidden,
        grid=grid,
        sub_hidden=sub_hidden,
        dropout=args.dropout,
        lambda1=args.lambda1,
        lambda2=args.lambda2,
        device=device,
    ).to(device)

    epoch = args.epochs
    pred_loss_train_history = []
    total_loss_train_history = []
    loss_valid_history = []
    optimizer = Adam(model.parameters(), lr=args.lr)
    compute_loss = torch.nn.MSELoss()
    min_valid_loss = sys.maxsize
    start_time = time.time()

    with tempfile.TemporaryDirectory(prefix=f"adafnn_seed_{args.seed}_") as tmp_dir:
        folder = tmp_dir + "/"
        Path(folder).mkdir(parents=True, exist_ok=True)
        best_epoch = 0

        for k in range(epoch):
            pred_loss_train = []
            total_loss_train = []
            loss_valid = []
            data.shuffle()
            model.train()

            for i, (x, y) in enumerate(data.get_train_batch()):
                x, y = x.to(device), y.to(device)
                out = model.forward(x)
                loss_pred = compute_loss(out, y)
                loss = (
                    loss_pred
                    + model.orthogonality_penalty(args.orth_pairs)
                    + model.sparsity_penalty(args.sparse_bases)
                )
                total_loss_train.append(loss.item())
                pred_loss_train.append(loss_pred.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss_train_history.append(np.mean(total_loss_train))
            pred_loss_train_history.append(np.mean(pred_loss_train))

            with torch.no_grad():
                model.eval()
                for x, y in data.get_valid_batch():
                    x, y = x.to(device), y.to(device)
                    valid_y = model.forward(x)
                    valid_loss = compute_loss(valid_y, y)
                    loss_valid.append(valid_loss.item())

            if np.mean(loss_valid) < min_valid_loss:
                save_adafnn_model(folder, "best", n_base, base_hidden, grid, sub_hidden, args.dropout, args.lambda1, args.lambda2, model, optimizer)
                min_valid_loss = np.mean(loss_valid)
                best_epoch = k + 1

            loss_valid_history.append(np.mean(loss_valid))

            if (k+1) % 50 == 0:
                print("epoch:", k+1, "\n",
                      "prediction training loss = ", pred_loss_train_history[-1],
                      "validation loss = ", loss_valid_history[-1])

        ck = folder + "best_checkpoint.pth"
        model, t = load_adafnn_model(ck, device)
        T = torch.tensor(t).to(device)
        t = np.array(t)

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
        "lambda1": args.lambda1,
        "lambda2": args.lambda2,
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
        "n_base": n_base,
        "base_hidden": ",".join(map(str, base_hidden)),
        "sub_hidden": ",".join(map(str, sub_hidden)),
        "dropout": args.dropout,
        "orth_pairs": args.orth_pairs,
        "sparse_bases": args.sparse_bases,
        "split_seed": args.split_seed,
        "device": str(device),
    }

    output_root = args.output_root or f"results/case{args.case}_mc"
    combo_dir = (
        Path(output_root)
        / f"adafnn_l1_{tag_float(args.lambda1)}_l2_{tag_float(args.lambda2)}"
    )
    csv_path = combo_dir / "mse.csv"
    write_or_replace_row(csv_path, row, CSV_FIELDS)

    print(
        f"wrote seed {args.seed} to {csv_path}; "
        f"test_mse={test_mse:.8g}, best_valid_mse={min_valid_loss:.8g}",
        flush=True,
    )
    return row


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=int, choices=[1, 2, 3, 4], default=1)
    parser.add_argument("--output-root", default="")
    parser.add_argument("--device", default="cpu", help="auto, cpu, cuda, cuda:0, ...")

    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--n-samples", type=int, default=4000)
    parser.add_argument("--n-grid", type=int, default=51)
    parser.add_argument("--me", type=float, default=None)
    parser.add_argument("--err", type=float, default=None)
    parser.add_argument("--split", type=int, nargs=3, default=[64, 16, 20])

    parser.add_argument("--n-base", type=int, default=0)
    parser.add_argument("--base-hidden", default="128,128,128")
    parser.add_argument("--sub-hidden", default="128,128,128")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lambda1", type=float, required=True)
    parser.add_argument("--lambda2", type=float, required=True)
    parser.add_argument("--orth-pairs", type=int, default=3)
    parser.add_argument("--sparse-bases", type=int, default=2)

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
