"""Обучение простого MLP на numpy: расписания learning rate, SGD с моментом,
ранняя остановка и чекпоинты.

Используется в experiments/08_numpy_mlp.ipynb как «ручная» альтернатива
sklearn.neural_network.MLPClassifier из 04_digits_augmentation.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# ------------------------------------------------------------------ activations and loss


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(x.dtype)


def softmax(z: np.ndarray) -> np.ndarray:
    """Численно устойчивый softmax по строкам (вычитается максимум строки)."""
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def cross_entropy(probs: np.ndarray, y: np.ndarray) -> float:
    """Средняя кросс-энтропия; y — целочисленные метки классов."""
    eps = 1e-12
    return float(-np.log(probs[np.arange(len(y)), y] + eps).mean())


# ------------------------------------------------------------------ learning-rate schedules


def constant_lr(base_lr: float, epoch: int, total_epochs: int) -> float:
    return base_lr


def step_lr(base_lr: float, epoch: int, total_epochs: int, step_size: int = 10, gamma: float = 0.5) -> float:
    """Ступенчатое затухание: LR умножается на gamma каждые step_size эпох."""
    return base_lr * gamma ** (epoch // step_size)


def cosine_lr(base_lr: float, epoch: int, total_epochs: int, min_lr: float = 1e-5, warmup_epochs: int = 3) -> float:
    """Косинусное затухание с линейным прогревом.

    Первые warmup_epochs эпох LR растёт линейно до base_lr, затем
    lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(pi * t)),
    где t — доля пройденных эпох после прогрева.
    """
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    t = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * t))


SCHEDULES = {"constant": constant_lr, "step": step_lr, "cosine": cosine_lr}

# ------------------------------------------------------------------ model


class NumpyMLP:
    """Перцептрон с одним скрытым слоем: Linear -> ReLU -> Linear -> softmax.

    Веса инициализируются по He (нормальное распределение с дисперсией 2 / fan_in),
    смещения — нулями.
    """

    def __init__(self, n_in: int, n_hidden: int, n_out: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0.0, math.sqrt(2.0 / n_in), (n_in, n_hidden))
        self.b1 = np.zeros(n_hidden)
        self.W2 = rng.normal(0.0, math.sqrt(2.0 / n_hidden), (n_hidden, n_out))
        self.b2 = np.zeros(n_out)
        self._cache = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        h_pre = X @ self.W1 + self.b1
        h = relu(h_pre)
        probs = softmax(h @ self.W2 + self.b2)
        self._cache = (X, h_pre, h, probs)
        return probs

    def backward(self, y: np.ndarray, weight_decay: float = 0.0) -> dict[str, np.ndarray]:
        """Градиенты кросс-энтропии по параметрам для последнего forward; L2 добавляется к весам, не к смещениям."""
        X, h_pre, h, probs = self._cache
        n = len(y)
        d_logits = probs.copy()
        d_logits[np.arange(n), y] -= 1.0
        d_logits /= n
        grads = {"W2": h.T @ d_logits + weight_decay * self.W2, "b2": d_logits.sum(axis=0)}
        d_h = (d_logits @ self.W2.T) * relu_grad(h_pre)
        grads["W1"] = X.T @ d_h + weight_decay * self.W1
        grads["b1"] = d_h.sum(axis=0)
        return grads

    def params(self) -> dict[str, np.ndarray]:
        return {"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2}

    def set_params(self, params: dict[str, np.ndarray]) -> None:
        for name, value in params.items():
            setattr(self, name, value.copy())

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X).argmax(axis=1)


# ------------------------------------------------------------------ optimiser


class SGDMomentum:
    """SGD с моментом (heavy ball): v = momentum * v - lr * g; p = p + v."""

    def __init__(self, params: dict[str, np.ndarray], momentum: float = 0.9):
        self.momentum = momentum
        self.velocity = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray], lr: float) -> None:
        for k in params:
            self.velocity[k] = self.momentum * self.velocity[k] - lr * grads[k]
            params[k] += self.velocity[k]  # in place: the arrays belong to the model


# ------------------------------------------------------------------ early stopping and history


@dataclass
class EarlyStopping:
    """Останавливает обучение, если метрика на валидации не улучшалась patience эпох подряд.

    Улучшением считается рост больше чем на min_delta. По умолчанию patience = 5.
    """

    patience: int = 5
    min_delta: float = 1e-4
    best: float = -math.inf
    best_epoch: int = -1
    bad_epochs: int = 0

    def update(self, value: float, epoch: int) -> bool:
        """Учесть метрику эпохи; вернуть True, если пора остановиться."""
        if value > self.best + self.min_delta:
            self.best, self.best_epoch, self.bad_epochs = value, epoch, 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience


@dataclass
class History:
    train_loss: list = field(default_factory=list)
    val_loss: list = field(default_factory=list)
    val_accuracy: list = field(default_factory=list)
    lr: list = field(default_factory=list)

    def best_epoch(self) -> int:
        return int(np.argmax(self.val_accuracy))


# ------------------------------------------------------------------ checkpoints


def save_checkpoint(model: NumpyMLP, path: str | Path, **meta) -> Path:
    """Сохранить веса в .npz, а метаданные (эпоха, метрика) — в JSON рядом с тем же именем."""
    path = Path(path).with_suffix(".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **model.params())
    path.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_checkpoint(model: NumpyMLP, path: str | Path) -> dict:
    """Загрузить веса из .npz в модель и вернуть метаданные из JSON (если есть)."""
    path = Path(path).with_suffix(".npz")
    data = np.load(path)
    model.set_params({k: data[k] for k in ("W1", "b1", "W2", "b2")})
    meta_path = path.with_suffix(".json")
    return json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}


# ------------------------------------------------------------------ training loop


class Trainer:
    """Обучение NumpyMLP мини-батчами с расписанием LR, weight decay и ранней остановкой.

    После обучения в модель возвращаются веса лучшей по val accuracy эпохи.
    """

    def __init__(
        self,
        model: NumpyMLP,
        lr: float = 0.05,
        epochs: int = 60,
        batch_size: int = 32,
        schedule: str = "cosine",
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        patience: int = 8,
        seed: int = 0,
        checkpoint_path: str | None = None,
        verbose: bool = False,
    ):
        if schedule not in SCHEDULES:
            raise ValueError(f"unknown schedule {schedule!r}, expected one of {sorted(SCHEDULES)}")
        self.model = model
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.schedule = schedule
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.patience = patience
        self.seed = seed
        self.checkpoint_path = checkpoint_path
        self.verbose = verbose
        self.stopped_epoch: int | None = None

    def _run_epoch(self, X: np.ndarray, y: np.ndarray, opt: SGDMomentum, lr: float, rng) -> float:
        order = rng.permutation(len(X))
        losses = []
        for start in range(0, len(order), self.batch_size):
            idx = order[start : start + self.batch_size]
            probs = self.model.forward(X[idx])
            losses.append(cross_entropy(probs, y[idx]))
            grads = self.model.backward(y[idx], self.weight_decay)
            opt.step(self.model.params(), grads, lr)
        return float(np.mean(losses))

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> History:
        rng = np.random.default_rng(self.seed)
        schedule = SCHEDULES[self.schedule]
        opt = SGDMomentum(self.model.params(), self.momentum)
        stopper = EarlyStopping(patience=self.patience)
        history = History()
        best_params = None
        for epoch in range(self.epochs):
            lr = schedule(self.lr, epoch, self.epochs)
            train_loss = self._run_epoch(X_train, y_train, opt, lr, rng)
            val_probs = self.model.predict_proba(X_val)
            val_acc = float((val_probs.argmax(axis=1) == y_val).mean())
            history.train_loss.append(train_loss)
            history.val_loss.append(cross_entropy(val_probs, y_val))
            history.val_accuracy.append(val_acc)
            history.lr.append(lr)
            stop = stopper.update(val_acc, epoch)
            if stopper.best_epoch == epoch:
                best_params = {k: v.copy() for k, v in self.model.params().items()}
                if self.checkpoint_path:
                    save_checkpoint(self.model, self.checkpoint_path, epoch=epoch, val_accuracy=val_acc)
            if self.verbose and (epoch % 10 == 0 or stop):
                print(f"epoch {epoch:3d}  lr={lr:.4f}  train_loss={train_loss:.4f}  val_acc={val_acc:.4f}")
            if stop:
                break
        self.stopped_epoch = epoch
        if best_params is not None:
            self.model.set_params(best_params)
        return history
