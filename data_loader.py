# data_loader.py
import os
import random
from glob import glob

import numpy as np
import torch
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split

from utils import DEVICE

# ------------------------------------------------------------
# Medical-safe transforms (KANSAS + HAM10000)
# ------------------------------------------------------------
def get_medical_transforms(img_size: int):
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10, fill=0),
        transforms.ToTensor(),
    ])

    test_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    strong_aug = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15, fill=0),
        transforms.ToTensor(),
    ])

    return train_tf, test_tf, strong_aug


# ------------------------------------------------------------
# Oversampling dataset wrapper
# ------------------------------------------------------------
class AugmentedImageDataset(Dataset):
    """
    Wraps an ImageFolder dataset and adds oversampled entries.
    Oversampled entries are reloaded and augmented with `aug_transform`.
    """
    def __init__(self, base_dataset, oversample_indices, aug_transform):
        self.base = base_dataset
        self.oversample = oversample_indices
        self.aug = aug_transform
        self.classes = base_dataset.classes

    def __len__(self):
        return len(self.base) + len(self.oversample)

    def __getitem__(self, idx):
        if idx < len(self.base):
            return self.base[idx]

        base_idx = self.oversample[idx - len(self.base)]
        path, label = self.base.samples[base_idx]
        img = Image.open(path).convert("RGB")
        img = self.aug(img)
        return img, label


def balance_dataset(dataset, aug_transform):
    from collections import Counter

    labels = [lbl for _, lbl in dataset.samples]
    counts = Counter(labels)
    print("Before balancing:", counts)

    max_count = max(counts.values())
    oversample_indices = []

    for cls, count in counts.items():
        needed = max_count - count
        if needed > 0:
            print(f" → Oversampling {dataset.classes[cls]} by {needed}")
            cls_indices = [i for i, (_, lbl) in enumerate(dataset.samples) if lbl == cls]
            oversample_indices += random.choices(cls_indices, k=needed)

    balanced_dataset = AugmentedImageDataset(dataset, oversample_indices, aug_transform)

    new_labels = [lbl for _, lbl in dataset.samples] + \
                 [dataset.samples[i][1] for i in oversample_indices]

    from collections import Counter as C2
    print("After balancing:", C2(new_labels))

    return balanced_dataset


# ------------------------------------------------------------
# HAM10000 preparation
# ------------------------------------------------------------
def prepare_ham10000(local_path="..."):
    import shutil
    import pandas as pd

    WORK = os.path.abspath(local_path)
    RAW = os.path.join(WORK, "raw_images")
    TRAIN = os.path.join(WORK, "train")
    TEST = os.path.join(WORK, "test")

    if os.path.exists(TRAIN) and os.path.exists(TEST):
        meta = pd.read_csv(os.path.join(WORK, "HAM10000_metadata.csv"))
        return TRAIN, TEST, sorted(meta["dx"].unique())

    os.makedirs(RAW, exist_ok=True)

    parts = [
        "HAM10000_images_part_1", "HAM10000_images_part_2",
        "ham10000_images_part_1", "ham10000_images_part_2",
        "ALL_IMAGES"
    ]

    for p in parts:
        folder = os.path.join(WORK, p)
        if os.path.exists(folder):
            for f in glob(os.path.join(folder, "*.jpg")):
                dst = os.path.join(RAW, os.path.basename(f))
                if not os.path.exists(dst):
                    shutil.copy(f, dst)

    import pandas as pd
    meta = pd.read_csv(os.path.join(WORK, "HAM10000_metadata.csv"))
    meta["filename"] = meta["image_id"].astype(str) + ".jpg"
    meta = meta[meta["filename"].isin(os.listdir(RAW))]

    classes = sorted(meta["dx"].unique())
    tr, te = train_test_split(meta, test_size=0.2, stratify=meta["dx"], random_state=42)

    for base in [TRAIN, TEST]:
        os.makedirs(base, exist_ok=True)
        for c in classes:
            os.makedirs(os.path.join(base, c), exist_ok=True)

    def cp(df, dest):
        for _, r in df.iterrows():
            src = os.path.join(RAW, r["filename"])
            dst = os.path.join(dest, r["dx"], r["filename"])
            shutil.copy(src, dst)

    cp(tr, TRAIN)
    cp(te, TEST)
    return TRAIN, TEST, classes


# ------------------------------------------------------------
# Unified dataloader
# ------------------------------------------------------------
def get_dataloaders(name: str,
                    batch_size: int = 16,
                    img_size: int = 224,
                    root_kansas: str = r"...",
                    root_ham: str = r"..."):

    name = name.upper()

    # CIFAR10 -------------------------------------------------
    if name == "CIFAR10":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor()
        ])

        train_ds = datasets.CIFAR10("./data", train=True, download=True, transform=transform_train)
        test_ds = datasets.CIFAR10("./data", train=False, download=True, transform=transform_test)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
        return train_loader, None, test_loader, train_ds.classes, None

    # HAM10000 ------------------------------------------------
    if name == "HAM10000":
        train_tf, test_tf, strong_aug = get_medical_transforms(img_size)
        TR, TE, classes = prepare_ham10000(root_ham)

        train_ds = datasets.ImageFolder(TR, transform=train_tf)
        test_ds = datasets.ImageFolder(TE, transform=test_tf)

        train_ds = balance_dataset(train_ds, strong_aug)

        labels = [lbl for _, lbl in train_ds]
        cw = compute_class_weight("balanced", classes=np.arange(len(classes)), y=labels)
        cw = torch.tensor(cw, dtype=torch.float32).to(DEVICE)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
        return train_loader, None, test_loader, classes, cw

    # KANSAS --------------------------------------------------
    if name == "KANSAS":
        train_tf, test_tf, strong_aug = get_medical_transforms(img_size)

        train_ds = datasets.ImageFolder(os.path.join(root_kansas, "train"), transform=train_tf)
        val_ds   = datasets.ImageFolder(os.path.join(root_kansas, "valid"), transform=test_tf)
        test_ds  = datasets.ImageFolder(os.path.join(root_kansas, "test"),  transform=test_tf)

        classes = train_ds.classes

        train_ds = balance_dataset(train_ds, strong_aug)

        labels = [lbl for _, lbl in train_ds]
        cw = compute_class_weight("balanced", classes=np.arange(len(classes)), y=labels)
        cw = torch.tensor(cw, dtype=torch.float32).to(DEVICE)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds,  batch_size=batch_size, shuffle=False)
        test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader, test_loader, classes, cw

    raise ValueError(f"Unknown dataset: {name}")
