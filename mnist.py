import code
import os
from typing import Literal

# from jaxtyping import UInt8
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

Split = Literal["train", "val", "test"]

SPLITS: dict[Split, str] = {
    "train": "data/mnist/resplit/mnist_train.csv",
    "val": "data/mnist/resplit/mnist_val.csv",
    "test": "data/mnist/original/mnist_test.csv",
}


# NOTE: Ruff doesn't like this return type, TODO figure out jaxtyping...
# tuple[UInt8[torch.Tensor, "n"], UInt8[torch.Tensor, "n d"]]
def load(split: Split) -> tuple[torch.Tensor, torch.Tensor]:
    csv_path = SPLITS[split]
    tensor_path = ".".join(csv_path.split(".")[:-1]) + ".pt"
    if not os.path.exists(tensor_path):
        data = np.loadtxt(csv_path, delimiter=",", dtype=np.uint8)
        t = torch.from_numpy(data)
        print(f"Saving to {tensor_path}")
        torch.save(t, tensor_path)
    else:
        t = torch.load(tensor_path)

    labels, features = t[:, 0], t[:, 1:]
    return labels, features


class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(784, 10)

    def forward(self, x):
        return self.linear(x)


def main() -> None:
    # Notes:
    # - manual optimizer (GD), manual data loading
    # - no shuffling, so this is batched GD rather than SGD
    # - the computation graph is freed after .backward() completes
    # - zero_grad(set_to_none=True) actually frees the gradients rather than setting
    #   values to 0. benefits on a dense model seem likely negligible. could imagine
    #   more free memory could allow larger val batch sizes.
    # - people typically zero_grad() at the start of the training loop

    epochs = 20
    batch_size = 100
    learning_rate = 0.1

    train_labels, train_features_raw = load("train")
    val_labels, val_features_raw = load("val")
    train_features = train_features_raw.float() / 255.0  # 0--255 -> 0--1
    val_features = val_features_raw.float() / 255.0
    model = LinearModel()
    loss_fn = nn.CrossEntropyLoss()

    with torch.no_grad():
        initial_predictions = model(val_features)
        correct = (initial_predictions.max(1).indices == val_labels).sum().item()
        total = val_labels.size(0)
        print(f"Starting val accuracy: {correct / total} ({correct}/{total})")

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        epoch_correct = 0
        print(f"Starting epoch {epoch}/{epochs}")
        for start in range(0, train_labels.size(0), batch_size):
            labels = train_labels[start : start + batch_size]  # (bsz)
            features = train_features[start : start + batch_size]  # (bsz,d)
            predictions = model(features)
            # no detach needed as .item() grabs float
            epoch_correct += (predictions.max(1).indices == labels).sum().item()
            loss = loss_fn(predictions, labels)
            loss.backward()
            epoch_loss += loss.item()

            with torch.no_grad():
                for param in model.parameters():
                    assert param.grad is not None  # for type checker
                    param -= learning_rate * param.grad

            model.zero_grad()

        # report train accuracy
        print(
            f"  Train acc during epoch {epoch}: {epoch_correct / train_labels.size(0)} ({epoch_correct}/{train_labels.size(0)})"
        )

        # compute validation accuracy
        with torch.no_grad():
            val_pred = model(val_features)
            correct = (val_pred.max(1).indices == val_labels).sum().item()
            total = val_labels.size(0)
            print(
                f"  Val acc after epoch {epoch}: {correct / total} ({correct}/{total})"
            )

    code.interact(local=dict(globals(), **locals()))


if __name__ == "__main__":
    main()
