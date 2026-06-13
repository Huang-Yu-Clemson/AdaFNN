"""Common code for AdaFNN simulation scripts."""

from __future__ import annotations

import csv
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.interpolate import BSpline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class LayerNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.d = d
        self.eps = eps
        self.alpha = nn.Parameter(torch.randn(d))
        self.beta = nn.Parameter(torch.randn(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True) + self.eps
        return (x - avg) / std * self.alpha + self.beta


class FeedForward(nn.Module):
    def __init__(
        self,
        in_d: int = 1,
        hidden: list[int] = [128, 128, 128],
        dropout: float = 0.0,
        activation=F.relu,
    ):
        super().__init__()
        self.sigma = activation
        dim = [in_d] + hidden + [1]
        self.layers = nn.ModuleList(
            [nn.Linear(dim[i - 1], dim[i]) for i in range(1, len(dim))]
        )
        self.ln = nn.ModuleList([LayerNorm(k) for k in hidden])
        self.dp = nn.ModuleList([nn.Dropout(dropout) for _ in range(len(hidden))])

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        for i in range(len(self.layers) - 1):
            t = self.layers[i](t)
            t = t + self.ln[i](t)
            t = self.sigma(t)
            t = self.dp[i](t)
        return self.layers[-1](t)


def inner_product(f1: torch.Tensor, f2: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    prod = f1 * f2
    return torch.matmul((prod[:, :-1] + prod[:, 1:]), h.unsqueeze(dim=-1)) / 2


def l1_norm(f: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    b, j = f.size()
    return inner_product(torch.abs(f), torch.ones((b, j), device=f.device), h)


def l2_norm(f: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(inner_product(f, f, h))


class AdaFNN(nn.Module):
    def __init__(
        self,
        n_base: int = 4,
        base_hidden: list[int] = [128, 128, 128],
        grid: np.ndarray | list[float] = (0.0, 1.0),
        sub_hidden: list[int] = [128, 128, 128],
        dropout: float = 0.0,
        lambda1: float = 0.0,
        lambda2: float = 0.0,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.n_base = n_base
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.device = device

        grid = np.array(grid)
        self.t = torch.tensor(grid).to(device).float()
        self.h = torch.tensor(grid[1:] - grid[:-1]).to(device).float()

        self.BL = nn.ModuleList(
            [
                FeedForward(
                    1,
                    hidden=base_hidden,
                    dropout=dropout,
                    activation=F.selu,
                )
                for _ in range(n_base)
            ]
        )
        self.FF = FeedForward(n_base, sub_hidden, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, J = x.size()
        assert J == self.h.size()[0] + 1

        T = self.t.unsqueeze(dim=-1)
        self.bases = [basis(T).transpose(-1, -2) for basis in self.BL]

        l2_norms = l2_norm(torch.cat(self.bases, dim=0), self.h).detach()
        self.normalized_bases = [
            self.bases[i] / (l2_norms[i, 0] + 1e-6) for i in range(self.n_base)
        ]

        score = torch.cat(
            [
                inner_product(b.repeat((B, 1)), x, self.h)
                for b in self.bases
            ],
            dim=-1,
        )
        out = self.FF(score)
        return out

    def orthogonality_penalty(self, n_pairs: int) -> torch.Tensor:
        """Paper lambda1: penalize non-orthogonality between learned bases."""
        if self.lambda1 == 0.0 or self.n_base == 1:
            return torch.zeros(1, device=self.device)

        max_pairs = self.n_base * (self.n_base - 1) // 2
        n_pairs = min(n_pairs, max_pairs)
        f1, f2 = [], []
        for _ in range(n_pairs):
            a, b = np.random.choice(self.n_base, 2, replace=False)
            f1.append(self.normalized_bases[a])
            f2.append(self.normalized_bases[b])

        return self.lambda1 * torch.mean(
            torch.abs(inner_product(torch.cat(f1, dim=0), torch.cat(f2, dim=0), self.h))
        )

    def sparsity_penalty(self, n_bases: int) -> torch.Tensor:
        """Paper lambda2: L1 sparsity penalty on learned bases."""
        if self.lambda2 == 0.0:
            return torch.zeros(1, device=self.device)

        selected = np.random.choice(self.n_base, min(n_bases, self.n_base), replace=False)
        selected_bases = torch.cat(
            [self.normalized_bases[i] for i in selected],
            dim=0,
        )
        return self.lambda2 * torch.mean(l1_norm(selected_bases, self.h))


z1 = [20, 5, 5] + [1] * 47
z2 = [1] * 50
z2[0] = z2[2] = 5
z2[4] = z2[9] = 3
Z = [z1, z2, z2, [1] * 50]


def paper_measurement_error_sd(case: int) -> float:
    if case in (3, 4):
        return float(np.sqrt(np.sum(np.array(Z[case - 1]) ** 2) / 10.0))
    return 0.0


def default_response_error_sd(case: int) -> float:
    if case in (3, 4):
        return 0.2
    return 0.0


def phi(k: int):
    if k == 1:
        return lambda t: np.ones((len(t),))
    return lambda t: np.sqrt(2) * np.cos((k - 1) * np.pi * t)


def _b1(t: np.ndarray) -> np.ndarray:
    return (4 - 16 * t) * (0 <= t) * (t <= 1/4)


def _b2(t: np.ndarray) -> np.ndarray:
    return (4 - 16 * np.abs(1/2 - t)) * (1/4 <= t) * (t <= 3/4)


class DataGenerator:
    def __init__(self, grid: np.ndarray, case: int = 1, me: float = 0.0, err: float = 0.0):
        if case < 1 or case > 4:
            raise ValueError(f"Simulation case must be 1, 2, 3, or 4; got {case}.")

        self.t = np.array(grid)
        self.me = me
        self.err = err
        self.case = case
        self.z = np.array(Z[case - 1])

    def generate(self, n: int = 1000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        r = np.random.uniform(low=-np.sqrt(3), high=np.sqrt(3), size=(n, 50))
        c = r * self.z
        basis = np.array([phi(k)(self.t) for k in range(1, 51)])
        x = np.matmul(c, basis)
        y = np.zeros((n, 1))

        if self.case == 1:
            y = (c[:, 2]) ** 2
        elif self.case == 4:
            beta1 = _b1(self.t)
            beta2 = _b2(self.t)
            h = np.array(self.t[1:] - self.t[:-1]).T
            for i in range(n):
                y[i, 0] = self._inner_product(beta2, x[i, :], h) + self._inner_product(beta1, x[i, :], h) ** 2
        else:
            y = (c[:, 4]) ** 2

        self.X = x + np.random.normal(0, self.me, size=(n, len(self.t)))
        self.Y = y.reshape((n, 1)) + np.random.normal(0, self.err, size=(n, 1))
        return self.X, self.Y, self.t

    def _inner_product(self, f1, f2, h):
        prod = f1 * f2
        if len(prod.shape) < 2:
            prod = prod.reshape((1, -1))
        res = np.matmul(prod[:, :-1] + prod[:, 1:], h) / 2
        return res


class SplitData:
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        t: np.ndarray,
        batch_size: int = 128,
        split: tuple[int, int, int] = (64, 16, 20),
        seed: int = 10294,
    ):
        self.t = t
        self.batch_size = batch_size

        n = x.shape[0]
        train_n = n // sum(split) * split[0]
        valid_n = n // sum(split) * split[1]
        test_n = n - train_n - valid_n
        self.train_B = train_n // batch_size
        self.valid_B = valid_n // batch_size
        self.test_B = test_n // batch_size

        np.random.seed(seed)
        order = list(range(n))
        np.random.shuffle(order)
        x = x[order, :]
        y = y[order, :]

        self.x_standardizer = StandardScaler()
        self.y_standardizer = StandardScaler()

        self.train_x = x[: (self.train_B * self.batch_size), :]
        self.train_y = y[: (self.train_B * self.batch_size), :]
        self.x_standardizer.fit(self.train_x)
        self.y_standardizer.fit(self.train_y)
        self.train_x = self.x_standardizer.transform(self.train_x)
        self.train_y = self.y_standardizer.transform(self.train_y)

        self.valid_x = x[
            (self.train_B * self.batch_size) : (
                (self.train_B + self.valid_B) * self.batch_size
            ),
            :,
        ]
        self.valid_y = y[
            (self.train_B * self.batch_size) : (
                (self.train_B + self.valid_B) * self.batch_size
            ),
            :,
        ]
        self.valid_x = self.x_standardizer.transform(self.valid_x)
        self.valid_y = self.y_standardizer.transform(self.valid_y)

        self.test_x = x[((self.train_B + self.valid_B) * self.batch_size) :, :]
        self.test_y = y[((self.train_B + self.valid_B) * self.batch_size) :, :]
        self.test_x = self.x_standardizer.transform(self.test_x)
        self.test_y = self.y_standardizer.transform(self.test_y)

    def shuffle(self):
        train_size = self.train_x.shape[0]
        new_order = list(range(train_size))
        np.random.shuffle(new_order)
        self.train_x = self.train_x[new_order, :]
        self.train_y = self.train_y[new_order, :]

    def shuffle_train(self) -> None:
        self.shuffle()

    def _batch_generator(self, X, Y, N):
        def generator_func():
            for i in range(1, N + 1):
                x = X[((i - 1) * self.batch_size):((i) * self.batch_size), :]
                y = Y[((i - 1) * self.batch_size):((i) * self.batch_size), :]

                yield torch.Tensor(x), torch.Tensor(y)

        return generator_func()

    def get_train_batch(self):
        return self._batch_generator(self.train_x, self.train_y, self.train_B)

    def get_valid_batch(self):
        return self._batch_generator(self.valid_x, self.valid_y, self.valid_B)

    def get_test_batch(self):
        return self._batch_generator(self.test_x, self.test_y, self.test_B)

    def batches(self, name: str, shuffle: bool = False):
        if name == "train":
            if shuffle:
                self.shuffle()
            return self.get_train_batch()
        elif name == "valid":
            return self.get_valid_batch()
        elif name == "test":
            return self.get_test_batch()
        raise ValueError(f"Unknown split: {name}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_int_list(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def save_checkpoint(path: Path, model: nn.Module, optimizer, config: dict) -> None:
    checkpoint = {
        "config": config,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, path)


def load_adafnn_checkpoint(path: Path, device: torch.device) -> AdaFNN:
    checkpoint = torch.load(path, map_location=device)
    config = checkpoint["config"]
    model = AdaFNN(
        n_base=config["n_base"],
        base_hidden=config["base_hidden"],
        grid=config["grid"],
        sub_hidden=config["sub_hidden"],
        dropout=config["dropout"],
        lambda1=config["lambda1"],
        lambda2=config["lambda2"],
        device=device,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    return model


def load_feedforward_checkpoint(path: Path, device: torch.device) -> FeedForward:
    checkpoint = torch.load(path, map_location=device)
    config = checkpoint["config"]
    model = FeedForward(
        in_d=config["input_dim"],
        hidden=config["hidden"],
        dropout=config["dropout"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    return model


def mse_on_split(
    model: nn.Module,
    data: SplitData,
    split_name: str,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    criterion = nn.MSELoss()
    losses = []
    pred_all = []
    true_all = []

    model.eval()
    with torch.no_grad():
        for x, y in data.batches(split_name):
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            losses.append(loss.item())
            pred_all.append(pred.detach().cpu().numpy())
            true_all.append(y.detach().cpu().numpy())

    return float(np.mean(losses)), np.vstack(pred_all), np.vstack(true_all)


def bspline_design(grid: np.ndarray, n_basis: int, degree: int = 3) -> np.ndarray:
    if n_basis <= degree:
        degree = n_basis - 1
    n_internal = n_basis - degree - 1
    internal = np.linspace(0.0, 1.0, n_internal + 2)[1:-1] if n_internal > 0 else []
    knots = np.r_[np.repeat(0.0, degree + 1), internal, np.repeat(1.0, degree + 1)]

    basis = []
    for j in range(n_basis):
        coef = np.zeros(n_basis)
        coef[j] = 1.0
        values = BSpline(knots, coef, degree, extrapolate=False)(grid)
        basis.append(np.nan_to_num(values))
    return np.vstack(basis).T


def bspline_scores(x: np.ndarray, grid: np.ndarray, n_basis: int) -> np.ndarray:
    design = bspline_design(grid, n_basis=n_basis)
    coef, *_ = np.linalg.lstsq(design, x.T, rcond=None)
    return coef.T


def fpca_scores(
    x: np.ndarray,
    split: tuple[int, int, int],
    split_seed: int,
    fve: float,
) -> tuple[np.ndarray, int, float]:
    """Fit PCA on the training split only and transform all observations."""
    n = x.shape[0]
    np.random.seed(split_seed)
    order = list(range(n))
    np.random.shuffle(order)
    train_n = n // sum(split) * split[0]
    train_x = x[order[:train_n]]

    pca_full = PCA().fit(train_x)
    cumulative = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = int(np.searchsorted(cumulative, fve) + 1)
    achieved_fve = float(cumulative[n_components - 1])

    pca = PCA(n_components=n_components).fit(train_x)
    scores = pca.transform(x)
    return scores, n_components, achieved_fve


def make_baseline_features(
    method: str,
    x: np.ndarray,
    grid: np.ndarray,
    split: tuple[int, int, int],
    split_seed: int,
    n_basis: int = 4,
    fve: float = 0.9,
) -> tuple[np.ndarray, dict]:
    if method == "raw":
        return x, {"input_dim": x.shape[1]}

    if method == "bspline":
        scores = bspline_scores(x, grid, n_basis)
        return scores, {"input_dim": n_basis, "n_basis": n_basis}

    if method == "fpca":
        scores, n_components, achieved_fve = fpca_scores(
            x,
            split=split,
            split_seed=split_seed,
            fve=fve,
        )
        return scores, {
            "input_dim": n_components,
            "target_fve": fve,
            "achieved_fve": achieved_fve,
        }

    raise ValueError(f"Unknown method: {method}")


def tag_float(value: float) -> str:
    return str(value).replace(".", "p")


def baseline_label(method: str, n_basis: int = 4, fve: float = 0.9) -> str:
    if method == "raw":
        return "raw_51"
    if method == "bspline":
        return f"bspline_{n_basis}"
    if method == "fpca":
        return f"fpca_{tag_float(fve)}"
    raise ValueError(f"Unknown method: {method}")


def write_or_replace_row(csv_path: Path, row: dict, fields: list[str]) -> None:
    """Safely write one row per seed to csv_path."""

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = csv_path.with_suffix(csv_path.suffix + ".lock")

    try:
        import fcntl  # Linux/Palmetto
    except ImportError:
        fcntl = None

    with lock_path.open("w", encoding="utf-8") as lock_file:
        if fcntl is not None:
            fcntl.flock(lock_file, fcntl.LOCK_EX)

        rows = []
        if csv_path.exists() and csv_path.stat().st_size > 0:
            with csv_path.open("r", newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))

        seed = str(row["seed"])
        rows = [old for old in rows if str(old.get("seed")) != seed]
        rows.append({field: row.get(field, "") for field in fields})
        rows.sort(key=lambda r: int(r["seed"]))

        tmp_path = csv_path.with_suffix(f".tmp.{os.getpid()}")
        with tmp_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, csv_path)

        if fcntl is not None:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def save_adafnn_model(folder, k, n_base, base_hidden, grid, sub_hidden, dropout, lambda1, lambda2, model, optimizer):
    checkpoint = {'n_base': n_base,
                  'base_hidden': base_hidden,
                  'grid': grid,
                  'sub_hidden': sub_hidden,
                  'dropout': dropout,
                  'lambda1' : lambda1,
                  'lambda2' : lambda2,
                  'state_dict': model.state_dict(),
                  'optimizer': optimizer.state_dict()}
    torch.save(checkpoint, folder + str(k) + '_' + 'checkpoint.pth')


def load_adafnn_model(file_path, device):
    checkpoint = torch.load(file_path)
    model = AdaFNN(n_base=checkpoint['n_base'],
                   base_hidden=checkpoint['base_hidden'],
                   grid=checkpoint['grid'],
                   sub_hidden=checkpoint['sub_hidden'],
                   dropout=checkpoint['dropout'],
                   lambda1=checkpoint['lambda1'],
                   lambda2=checkpoint['lambda2'],
                   device=device)
    model.load_state_dict(checkpoint['state_dict'])
    _ = model.to(device)
    return model, checkpoint['grid']


def save_feedforward_model(folder, k, input_dim, hidden, dropout, model, optimizer):
    checkpoint = {'input_dim': input_dim,
                  'hidden': hidden,
                  'dropout': dropout,
                  'state_dict': model.state_dict(),
                  'optimizer': optimizer.state_dict()}
    torch.save(checkpoint, folder + str(k) + '_' + 'checkpoint.pth')


def load_feedforward_model(file_path, device):
    checkpoint = torch.load(file_path)
    model = FeedForward(in_d=checkpoint['input_dim'],
                        hidden=checkpoint['hidden'],
                        dropout=checkpoint['dropout'])
    model.load_state_dict(checkpoint['state_dict'])
    _ = model.to(device)
    return model
