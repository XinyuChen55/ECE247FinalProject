from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def extract_scalar_from_events(event_files, tag):
    rows = []

    for f in event_files:
        ea = EventAccumulator(str(f))
        ea.Reload()

        scalar_tags = ea.Tags().get("scalars", [])
        if tag not in scalar_tags:
            continue

        for e in ea.Scalars(tag):
            rows.append(
                {
                    "file": f.name,
                    "wall_time": e.wall_time,
                    "step": e.step,
                    "value": e.value,
                }
            )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = (
            df.sort_values(["step", "wall_time"])
            .drop_duplicates(subset=["step"], keep="last")
            .reset_index(drop=True)
        )
    return df


def main():
    log_dir = Path("/home/xinyu001/emg2qwerty/logs/2026-03-12/best-40epoch/lightning_logs/version_0")
    out_dir = Path("/home/xinyu001/emg2qwerty/plots")
    out_dir.mkdir(exist_ok=True)

    event_files = sorted(log_dir.glob("events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event files found in: {log_dir}")

    print("Found event files:")
    for f in event_files:
        print("  ", f)

    all_tags = set()
    for f in event_files:
        ea = EventAccumulator(str(f))
        ea.Reload()
        tags = ea.Tags().get("scalars", [])
        all_tags.update(tags)

    print("\nAvailable scalar tags:")
    for tag in sorted(all_tags):
        print("  ", tag)

    train_loss_df = extract_scalar_from_events(event_files, "train/loss")
    val_loss_df = extract_scalar_from_events(event_files, "val/loss")
    train_cer_df = extract_scalar_from_events(event_files, "train/CER")
    val_cer_df = extract_scalar_from_events(event_files, "val/CER")

    steps_per_epoch = 120

    for df in [train_loss_df, val_loss_df, train_cer_df, val_cer_df]:
        if not df.empty:
            df["epoch"] = df["step"] / steps_per_epoch

    #Loss curve
    #start from epoch 1
    train_loss_plot = train_loss_df[train_loss_df["epoch"] >= 1]
    val_loss_plot = val_loss_df[val_loss_df["epoch"] >= 1]

    plt.figure(figsize=(8, 5))
    plt.plot(train_loss_plot["epoch"], train_loss_plot["value"], label="train loss")
    plt.plot(val_loss_plot["epoch"], val_loss_plot["value"], label="val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Curve (Epoch >= 1)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "loss_curve_1.png", dpi=300, bbox_inches="tight")

    plt.figure(figsize=(8, 5))
    if not train_loss_df.empty:
        plt.plot(train_loss_df["epoch"], train_loss_df["value"], label="train loss")
    if not val_loss_df.empty:
        plt.plot(val_loss_df["epoch"], val_loss_df["value"], label="val loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Curve")
    plt.legend()
    plt.tight_layout()
    loss_path = out_dir / "loss_curve.png"
    plt.savefig(loss_path, dpi=300, bbox_inches="tight")
    plt.close()

    #CER curve
    plt.figure(figsize=(8, 5))
    if not train_cer_df.empty:
        plt.plot(train_cer_df["epoch"], train_cer_df["value"], label="train CER")
    if not val_cer_df.empty:
        plt.plot(val_cer_df["epoch"], val_cer_df["value"], label="val CER")

    plt.xlabel("Epoch")
    plt.ylabel("CER")
    plt.title("Training and Validation CER Curve")
    plt.legend()
    plt.tight_layout()
    cer_path = out_dir / "cer_curve.png"
    plt.savefig(cer_path, dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()