from src.dataset import load_food101
from src.model import load_clip
from src.evaluate import build_prompts, evaluate_zero_shot_batched

from pathlib import Path
import csv
from datetime import datetime


def main():
    # Load dataset + labels
    ds, label_names = load_food101()

    model, processor, device = load_clip()

    prompts = build_prompts(label_names)

  
    top1, top5 = evaluate_zero_shot_batched(
        model=model,
        processor=processor,
        device=device,
        dataset_split=ds["validation"],
        prompts=prompts,
        num_samples=2000,   
        batch_size=32,      
    )

    print(f"Device: {device}")
    print(f"Top-1: {top1:.3f} | Top-5: {top5:.3f}")

    Path("results").mkdir(exist_ok=True)

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": "openai/clip-vit-base-patch32",
        "dataset": "food101",
        "num_samples": 2000,
        "device": device,
        "top1": top1,
        "top5": top5,
    }

    csv_path = Path("results/metrics.csv")
    write_header = not csv_path.exists()

    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print(f"Saved to {csv_path}")


if __name__ == "__main__":
    main()