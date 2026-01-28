# training.py

import time
import torch
import torch.nn as nn
import torch.optim as optim

from utils import DEVICE
from colorspace import convert_and_normalize


# ============================================================
#  EVALUATE MODEL
# ============================================================
def evaluate_model(model, loader, color_space="rgb"):
    """
    Computes classification accuracy (%) on a data loader.
    Used by train_model() for both train & test accuracy.
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(DEVICE), target.to(DEVICE)

            # Convert to target color space
            data_cs = convert_and_normalize(data, color_space)

            outputs = model(data_cs)
            preds = outputs.argmax(dim=1)

            correct += (preds == target).sum().item()
            total += target.size(0)

    return 100.0 * correct / max(total, 1)


# ============================================================
#  TRAIN MODEL
# ============================================================
def train_model(
    model,
    train_loader,
    test_loader,
    epochs=5,
    color_space="rgb",
    class_weights=None,
):
    model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    criterion = nn.CrossEntropyLoss(weight=class_weights) if class_weights is not None \
                else nn.CrossEntropyLoss()

    train_loss_history = []
    test_loss_history = []
    train_acc_history = []
    test_acc_history = []
    grad_norm_history = []
    epoch_times = []

    for epoch in range(epochs):
        start_t = time.time()
        model.train()
        running_train_loss = 0.0

        # =====================================================
        # TRAIN LOOP
        # =====================================================
        for data, target in train_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            data_cs = convert_and_normalize(data, color_space)

            optimizer.zero_grad()
            outputs = model(data_cs)
            loss = criterion(outputs, target)
            loss.backward()

            # gradient norm of the first convolution
            if hasattr(model, "get_first_conv"):
                first_conv = model.get_first_conv()
                grad_norm = 0.0
                for p in first_conv.parameters():
                    if p.grad is not None:
                        grad_norm += float(p.grad.norm().item())
                grad_norm_history.append(grad_norm)

            optimizer.step()
            running_train_loss += loss.item()

        avg_train_loss = running_train_loss / max(len(train_loader), 1)
        train_loss_history.append(avg_train_loss)

        # =====================================================
        # TEST LOSS
        # =====================================================
        model.eval()
        running_test_loss = 0.0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                data_cs = convert_and_normalize(data, color_space)
                outputs = model(data_cs)
                loss = criterion(outputs, target)
                running_test_loss += loss.item()

        avg_test_loss = running_test_loss / max(len(test_loader), 1)
        test_loss_history.append(avg_test_loss)

        # =====================================================
        # ACCURACY
        # =====================================================
        train_acc = evaluate_model(model, train_loader, color_space)
        test_acc = evaluate_model(model, test_loader, color_space)

        train_acc_history.append(train_acc)
        test_acc_history.append(test_acc)

        epoch_times.append(time.time() - start_t)

        print(
            f"[{color_space.upper()}] Epoch {epoch+1}/{epochs} "
            f"| Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f} "
            f"| Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%"
        )

    # ============================================================
    # RETURN ALL TRAINING STATISTICS FOR PLOTS + ANALYSIS
    # ============================================================
    return {
        "loss": train_loss_history,          # alias for backwards compatibility
        "train_loss": train_loss_history,
        "test_loss": test_loss_history,
        "train_acc": train_acc_history,
        "test_acc": test_acc_history,
        "grad": grad_norm_history,
        "epoch_times": epoch_times,
        "first_layer_activation": getattr(model, "first_layer_act", None),
    }


# ============================================================
# 🔥 TRAIN IN TWO STAGES
# ============================================================
def train_in_stages(
    model,
    train_loader,
    test_loader,
    first_stage_epochs=3,
    final_epochs=10,
    color_space="rgb",
    class_weights=None,
):
    print(f"\n=== Stage 1: Training for {first_stage_epochs} epochs ===")
    stats_stage1 = train_model(
        model,
        train_loader,
        test_loader,
        epochs=first_stage_epochs,
        color_space=color_space,
        class_weights=class_weights,
    )

    remaining_epochs = max(final_epochs - first_stage_epochs, 0)
    print(f"\n=== Stage 2: Continuing for {remaining_epochs} more epochs ===")
    stats_stage2 = train_model(
        model,
        train_loader,
        test_loader,
        epochs=remaining_epochs,
        color_space=color_space,
        class_weights=class_weights,
    )

    # Merge both stages into one continuous history
    merged = {}
    for key in stats_stage1:
        if isinstance(stats_stage1[key], list):
            merged[key] = stats_stage1[key] + stats_stage2[key]
        else:
            merged[key] = stats_stage2[key]

    return merged
