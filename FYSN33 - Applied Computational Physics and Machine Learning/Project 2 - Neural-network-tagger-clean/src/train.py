# src/train.py

import torch

def train_model(
    model,
    X_train_t,
    y_train_t,
    X_val_t,
    y_val_t,
    optimizer,
    loss_fn,
    max_epochs=200,
    batch_size=256,
    patience=15,
    min_delta=1e-4,
):

    train_losses = []
    val_losses = []

    best_val = float("inf")
    best_state = None
    epochs_no_improve = 0

    N = X_train_t.shape[0]

    for epoch in range(1, max_epochs + 1):

        # ---- Training ----
        model.train()
        perm = torch.randperm(N)
        total_train_loss = 0.0

        for i in range(0, N, batch_size):
            idx = perm[i:i+batch_size]
            xb = X_train_t[idx]
            yb = y_train_t[idx]

            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * xb.size(0)

        avg_train_loss = total_train_loss / N

        # ---- Validation ----
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_loss = loss_fn(val_logits, y_val_t).item()

        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)

        # ---- Early stopping ----
        if val_loss < best_val - min_delta:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | train loss {avg_train_loss:.4f} | val loss {val_loss:.4f} "
                f"| best {best_val:.4f} | no improve {epochs_no_improve}/{patience}")

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch} (best val loss = {best_val:.4f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, train_losses, val_losses