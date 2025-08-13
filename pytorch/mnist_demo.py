import torch
from torch import nn
import torchvision
from torchvision import transforms
import torch.nn.functional as F
import os, sys, time
import numpy as np
import pandas as pd
import datetime
from tqdm import tqdm
from copy import deepcopy

def printlog(info):
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("\n" + "=========="*8 + f"{nowtime}")
    print(str(info) + "\n")

class Accuracy(nn.Module):
    def __init__(self):
        super().__init__()
        self.correct = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.total = nn.Parameter(torch.tensor(0.0), requires_grad=False)

    def forward(self, preds: torch.Tensor, targets: torch.Tensor):
        preds = preds.argmax(dim=-1)
        m = (preds == targets).sum()
        n = targets.shape[0]
        self.correct += m
        self.total += n
        return m / n

    def compute(self):
        return self.correct.float() / self.total

    def reset(self):
        self.correct -= self.correct
        self.total -= self.total

def create_net():
    net = nn.Sequential()
    net.add_module("conv1", nn.Conv2d(1, 64, 3))
    net.add_module("pool1", nn.MaxPool2d(2, 2))
    net.add_module("conv2", nn.Conv2d(64, 512, 3))
    net.add_module("pool2", nn.MaxPool2d(2, 2))
    net.add_module("dropout", nn.Dropout2d(0.1))
    net.add_module("adaptive_pool", nn.AdaptiveMaxPool2d((1,1)))
    net.add_module("flatten", nn.Flatten())
    net.add_module("linear1", nn.Linear(512, 1024))
    net.add_module("relu", nn.ReLU())
    net.add_module("linear2", nn.Linear(1024, 10))
    return net

def main():
    #================================================================================
    # 一，准备数据
    #================================================================================
    transform = transforms.Compose([transforms.ToTensor()])
    ds_train = torchvision.datasets.MNIST(root="mnist/", train=True, download=True, transform=transform)
    ds_val = torchvision.datasets.MNIST(root="mnist/", train=False, download=True, transform=transform)

    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=128, shuffle=True, num_workers=2)
    dl_val = torch.utils.data.DataLoader(ds_val, batch_size=128, shuffle=False, num_workers=2)

    #================================================================================
    # 二，定义模型
    #================================================================================
    net = create_net()
    print(net)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    metrics_dict = nn.ModuleDict({"acc": Accuracy()})

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    net.to(device)
    loss_fn.to(device)
    metrics_dict.to(device)

    epochs = 20
    ckpt_path = 'checkpoint.pt'

    monitor = "val_acc"
    patience = 5
    mode = "max"

    history = {}

    for epoch in range(1, epochs + 1):
        printlog(f"Epoch {epoch} / {epochs}")

        # Train
        net.train()
        total_loss, step = 0, 0
        loop = tqdm(enumerate(dl_train), total=len(dl_train), ncols=100)
        train_metrics_dict = deepcopy(metrics_dict)

        for i, batch in loop:
            features, labels = batch
            features = features.to(device)
            labels = labels.to(device)

            preds = net(features)
            loss = loss_fn(preds, labels)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            step_metrics = {"train_" + name: metric_fn(preds, labels).item() for name, metric_fn in train_metrics_dict.items()}
            step_log = dict({"train_loss": loss.item()}, **step_metrics)

            total_loss += loss.item()
            step += 1

            if i != len(dl_train) - 1:
                loop.set_postfix(**step_log)
            else:
                epoch_loss = total_loss / step
                epoch_metrics = {"train_" + name: metric_fn.compute().item() for name, metric_fn in train_metrics_dict.items()}
                epoch_log = dict({"train_loss": epoch_loss}, **epoch_metrics)
                loop.set_postfix(**epoch_log)
                for name, metric_fn in train_metrics_dict.items():
                    metric_fn.reset()

        for name, metric in epoch_log.items():
            history[name] = history.get(name, []) + [metric]

        # Validation
        net.eval()
        total_loss, step = 0, 0
        loop = tqdm(enumerate(dl_val), total=len(dl_val), ncols=100)
        val_metrics_dict = deepcopy(metrics_dict)

        with torch.no_grad():
            for i, batch in loop:
                features, labels = batch
                features = features.to(device)
                labels = labels.to(device)

                preds = net(features)
                loss = loss_fn(preds, labels)

                step_metrics = {"val_" + name: metric_fn(preds, labels).item() for name, metric_fn in val_metrics_dict.items()}
                step_log = dict({"val_loss": loss.item()}, **step_metrics)

                total_loss += loss.item()
                step += 1

                if i != len(dl_val) - 1:
                    loop.set_postfix(**step_log)
                else:
                    epoch_loss = total_loss / step
                    epoch_metrics = {"val_" + name: metric_fn.compute().item() for name, metric_fn in val_metrics_dict.items()}
                    epoch_log = dict({"val_loss": epoch_loss}, **epoch_metrics)
                    loop.set_postfix(**epoch_log)
                    for name, metric_fn in val_metrics_dict.items():
                        metric_fn.reset()

        epoch_log["epoch"] = epoch
        for name, metric in epoch_log.items():
            history[name] = history.get(name, []) + [metric]

        # Early stopping
        arr_scores = history[monitor]
        best_score_idx = np.argmax(arr_scores) if mode == "max" else np.argmin(arr_scores)
        if best_score_idx == len(arr_scores) - 1:
            torch.save(net.state_dict(), ckpt_path)
            print(f"<<<<<< reach best {monitor} : {arr_scores[best_score_idx]} >>>>>>", file=sys.stderr)
        if len(arr_scores) - best_score_idx > patience:
            print(f"<<<<<< {monitor} without improvement in {patience} epoch, early stopping >>>>>>", file=sys.stderr)
            break

        net.load_state_dict(torch.load(ckpt_path))

    dfhistory = pd.DataFrame(history)
    print(dfhistory.tail())

if __name__ == "__main__":
    main()
