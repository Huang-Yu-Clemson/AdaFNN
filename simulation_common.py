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

    def __init__(self, d, eps=1e-6):
        super().__init__()
        # d is the normalization dimension
        self.d = d
        self.eps = eps
        self.alpha = nn.Parameter(torch.randn(d))
        self.beta = nn.Parameter(torch.randn(d))

    def forward(self, x):
        # x is a torch.Tensor
        # avg is the mean value of a layer
        avg = x.mean(dim=-1, keepdim=True)
        # std is the standard deviation of a layer (eps is added to prevent dividing by zero)
        std = x.std(dim=-1, keepdim=True) + self.eps
        return (x - avg) / std * self.alpha + self.beta


class FeedForward(nn.Module):

    def __init__(self, in_d=1, hidden=[4,4,4], dropout=0.1, activation=F.relu):
        # in_d      : input dimension, integer
        # hidden    : hidden layer dimension, array of integers
        # dropout   : dropout probability, a float between 0.0 and 1.0
        # activation: activation function at each layer
        super().__init__()
        self.sigma = activation
        dim = [in_d] + hidden + [1]
        self.layers = nn.ModuleList([nn.Linear(dim[i-1], dim[i]) for i in range(1, len(dim))])
        self.ln = nn.ModuleList([LayerNorm(k) for k in hidden])
        self.dp = nn.ModuleList([nn.Dropout(dropout) for _ in range(len(hidden))])

    def forward(self, t):
        for i in range(len(self.layers)-1):
            t = self.layers[i](t)
            # skipping connection
            t = t + self.ln[i](t)
            t = self.sigma(t)
            # apply dropout
            t = self.dp[i](t)
        # linear activation at the last layer
        return self.layers[-1](t)


def _inner_product(f1, f2, h):
    """    
    f1 - (B, J) : B functions, observed at J time points,
    f2 - (B, J) : same as f1
    h  - (J-1,1): weights used in the trapezoidal rule
    pay attention to dimension
    <f1, f2> = sum (h/2) (f1(t{j}) + f2(t{j+1}))
    """
    prod = f1 * f2 # (B, J = len(h) + 1)
    return torch.matmul((prod[:, :-1] + prod[:, 1:]), h.unsqueeze(dim=-1))/2


def _l1(f, h):
    # f dimension : ( B bases, J )
    B, J = f.size()
    return _inner_product(torch.abs(f), torch.ones((B, J)), h)


def _l2(f, h):
    # f dimension : ( B bases, J )
    # output dimension - ( B bases, 1 )
    return torch.sqrt(_inner_product(f, f, h)) 


class AdaFNN(nn.Module):

    def __init__(self, n_base=4, base_hidden=[64, 64, 64], grid=(0, 1),
                 sub_hidden=[128, 128, 128], dropout=0.1, lambda1=0.0, lambda2=0.0,
                 device=None):
        """
        n_base      : number of basis nodes, integer
        base_hidden : hidden layers used in each basis node, array of integers
        grid        : observation time grid, array of sorted floats including 0.0 and 1.0
        sub_hidden  : hidden layers in the subsequent network, array of integers
        dropout     : dropout probability
        lambda1     : penalty of L1 regularization, a positive real number
        lambda2     : penalty of L2 regularization, a positive real number
        device      : device for the training
        """
        super().__init__()
        self.n_base = n_base
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.device = device
        # grid should include both end points
        grid = np.array(grid)
        # send the time grid tensor to device
        self.t = torch.tensor(grid).to(device).float()
        self.h = torch.tensor(grid[1:] - grid[:-1]).to(device).float()
        # instantiate each basis node in the basis layer
        self.BL = nn.ModuleList([FeedForward(1, hidden=base_hidden, dropout=dropout, activation=F.selu)
                                 for _ in range(n_base)])
        # instantiate the subsequent network
        self.FF = FeedForward(n_base, sub_hidden, dropout)

    def forward(self, x):
        B, J = x.size()
        assert J == self.h.size()[0] + 1
        T = self.t.unsqueeze(dim=-1)
        # evaluate the current basis nodes at time grid
        self.bases = [basis(T).transpose(-1, -2) for basis in self.BL]
        """
        compute each basis node's L2 norm
        normalize basis nodes
        """
        l2_norm = _l2(torch.cat(self.bases, dim=0), self.h).detach()
        self.normalized_bases = [self.bases[i] / (l2_norm[i, 0] + 1e-6) for i in range(self.n_base)]
        # compute each score <basis_i, f> 
        score = torch.cat([_inner_product(b.repeat((B, 1)), x, self.h) # (B, 1)
                           for b in self.bases], dim=-1) # score dim = (B, n_base)
        # take the tensor of scores into the subsequent network
        out = self.FF(score)
        return out

    def R1(self, l2_pairs):
        """
        Orthogonality regularization in the paper.
        lambda1 controls the penalty strength.
        l2_pairs : number of pairs to regularize, integer
        """
        if self.lambda1 == 0 or self.n_base == 1: return torch.zeros(1).to(self.device)
        k = min(l2_pairs, self.n_base * (self.n_base - 1) // 2)
        f1, f2 = [None] * k, [None] * k
        for i in range(k):
            a, b = np.random.choice(self.n_base, 2, replace=False)
            f1[i], f2[i] = self.normalized_bases[a], self.normalized_bases[b]
        return self.lambda1 * torch.mean(torch.abs(_inner_product(torch.cat(f1, dim=0),
                                                                  torch.cat(f2, dim=0),
                                                                  self.h)))

    def R2(self, l1_k):
        """
        L1 sparsity regularization in the paper.
        lambda2 controls the penalty strength.
        l1_k : number of basis nodes to regularize, integer
        """
        if self.lambda2 == 0: return torch.zeros(1).to(self.device)
        selected = np.random.choice(self.n_base, min(l1_k, self.n_base), replace=False)
        selected_bases = torch.cat([self.normalized_bases[i] for i in selected], dim=0) # (k, J)
        return self.lambda2 * torch.mean(_l1(selected_bases, self.h))


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


def _phi(k):
    if k == 1: return lambda t: np.ones((len(t),))
    return lambda t : np.sqrt(2) * np.cos((k-1) * np.pi * t)

def _b1(t):
    return (4 - 16 * t) * (0 <= t) * (t <= 1/4)


def _b2(t):
    return (4 - 16 * np.abs(1/2 - t)) * (1/4 <= t) * (t <= 3/4)


class DataGenerator:

    def __init__(self, grid, case=1, me=1, err=1):
        """
        grid : array of time points, floats
        case : case number, integer
        me   : variance of measurement error added to X, non-negative real value
        err  : variance of noise added to Y, non-negative real value
        """
        self.t = np.array(grid)
        # measurement error
        self.me = me
        self.err = err
        # case - 1
        self.case = case
        self.z = np.array(Z[case-1])

    def generate(self, n=1000):
        """
        n : number of subjects to generate, integer
        """
        # X = sum c_k phi_k
        # c_k = z_k r_k, r_k iid unif[-sqrt(3), sqrt(3)]
        # generate r
        r = np.random.uniform(low=-np.sqrt(3), high=np.sqrt(3), size=(n, 50))
        c = r * self.z # (n, 50) elementwise multiplication
        phi = np.array([_phi(k)(self.t) for k in range(1, 51)]) # (50, len(self.t))
        X = np.matmul(c, phi) # (n, len(self.t))
        Y = np.zeros((n, 1))
        if self.case == 1:
            Y = (c[:, 2]) ** 2
        elif self.case == 4:
            beta1 = _b1(self.t)
            beta2 = _b2(self.t)
            h = np.array(self.t[1:] - self.t[:-1]).T
            for i in range(n):
                Y[i, 0] = self._inner_product(beta2, X[i, :], h) + self._inner_product(beta1, X[i, :], h) ** 2

        else: # self.case = 2 or 3
            Y = (c[:, 4]) ** 2        
        self.X = X + np.random.normal(0, self.me, size=(n, len(self.t)))
        self.Y = Y.reshape((n, 1)) + np.random.normal(0, self.err, size=(n, 1))
        return self.X, self.Y, self.t

    def _inner_product(self, f1, f2, h):
        prod = f1 * f2
        if len(prod.shape) < 2:
            prod = prod.reshape((1, -1))
        res = np.matmul(prod[:, :-1] + prod[:, 1:], h) / 2
        return res


class DataLoader:

    def __init__(self, batch_size, X, Y, T, split=(8, 1, 1), random_seed=10294):
        """
        batch_size : batch size, integer
        X - (n, J) : pandas.DataFrame for observed functional data, n - subject number, J - number of time points
        Y - (n, 1) : pandas.DataFrame for response
        split      : train/valid/test split
        random_seed: random seed for training data re-shuffle
        """        
        self.n, J = X.shape
        self.t = T.iloc[0, :].to_numpy()
        X, Y = X.values, Y.values

        # train/valid/test split
        self.batch_size = batch_size
        train_n = self.n // sum(split) * split[0]
        valid_n = self.n // sum(split) * split[1]
        test_n = self.n - train_n - valid_n
        self.train_B = train_n // batch_size
        self.valid_B = valid_n // batch_size
        self.test_B = test_n // batch_size

        # random shuffle
        np.random.seed(random_seed)
        _order = list(range(self.n))
        np.random.shuffle(_order)
        X = X[_order, :]
        Y = Y[_order, :]

        # standardize dataset based on the training dataset
        self.X_standardizer = StandardScaler()
        self.Y_standardizer = StandardScaler()

        # train/valid/test split
        self.train_X = X[:(self.train_B * self.batch_size), :]
        self.train_Y = Y[:(self.train_B * self.batch_size), :]
        self.X_standardizer.fit(self.train_X)
        self.Y_standardizer.fit(self.train_Y)
        self.train_X = self.X_standardizer.transform(self.train_X)
        self.train_Y = self.Y_standardizer.transform(self.train_Y)

        self.valid_X = X[(self.train_B * self.batch_size):((self.train_B + self.valid_B) * self.batch_size), :]
        self.valid_Y = Y[(self.train_B * self.batch_size):((self.train_B + self.valid_B) * self.batch_size), :]
        self.valid_X = self.X_standardizer.transform(self.valid_X)
        self.valid_Y = self.Y_standardizer.transform(self.valid_Y)

        self.test_X = X[((self.train_B + self.valid_B) * self.batch_size):, :]
        self.test_Y = Y[((self.train_B + self.valid_B) * self.batch_size):, :]
        self.test_X = self.X_standardizer.transform(self.test_X)
        self.test_Y = self.Y_standardizer.transform(self.test_Y)

    def shuffle(self):
        # re-shuffle the training dataset
        train_size = self.train_X.shape[0]
        new_order = list(range(train_size))
        np.random.shuffle(new_order)
        self.train_X = self.train_X[new_order, :]
        self.train_Y = self.train_Y[new_order, :]

    def _batch_generator(self, X, Y, N):

        def generator_func():
            for i in range(1, N):
                x = X[((i - 1) * self.batch_size):((i) * self.batch_size), :]
                y = Y[((i - 1) * self.batch_size):((i) * self.batch_size), :]

                yield torch.Tensor(x), torch.Tensor(y)

        return generator_func()

    def get_train_batch(self):
        return self._batch_generator(self.train_X, self.train_Y, self.train_B)

    def get_valid_batch(self):
        return self._batch_generator(self.valid_X, self.valid_Y, self.valid_B)

    def get_test_batch(self):
        return self._batch_generator(self.test_X, self.test_Y, self.test_B)


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
    data: DataLoader,
    split_name: str,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    criterion = nn.MSELoss()
    losses = []
    pred_all = []
    true_all = []
    if split_name == "train":
        batch_iter = data.get_train_batch()
    elif split_name == "valid":
        batch_iter = data.get_valid_batch()
    elif split_name == "test":
        batch_iter = data.get_test_batch()
    else:
        raise ValueError(f"Unknown split: {split_name}")

    model.eval()
    with torch.no_grad():
        for x, y in batch_iter:
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
