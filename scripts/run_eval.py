from src.dataset import load_food101
from src.model import load_clip
from src.evaluate import (
    build_prompts,
    evaluate_zero_shot_batched,
    collect_failures,
)

from pathlib import Path
import csv
from datetime import datetime


def log_result(row):
    Path("results").mkdir(exist_ok=True)
    csv_path = Path("results/metrics.csv")
    write_header = not csv_path.exists()

    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main():
    ds, label_names = load_food101()
    model, processor, device = load_clip()

    templates = [
        "a photo of {label}",
        "a centered photo of {label}",
        "a close-up photo of {label}",
    ]

    num_samples = 2000
    batch_size = 32

  
    for template in templates:
        prompts = build_prompts(label_names, template=template)

        top1, top5 = evaluate_zero_shot_batched(
            model=model,
            processor=processor,
            device=device,
            dataset_split=ds["validation"],
            prompts=prompts,
            num_samples=num_samples,
            batch_size=batch_size,
        )

        print(f"\nTemplate: {template}")
        print(f"Device: {device}")
        print(f"Top-1: {top1:.3f} | Top-5: {top5:.3f}")

        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "model": "openai/clip-vit-base-patch32",
            "dataset": "food101",
            "num_samples": num_samples,
            "batch_size": batch_size,
            "device": device,
            "template": template,
            "top1": top1,
            "top5": top5,
        }
        log_result(row)

    print("\nSaved all results to results/metrics.csv")

 
    print("\n[FAILURES] Collecting misclassified examples...")

    failure_template = "a close-up photo of {label}"
    failure_prompts = build_prompts(label_names, template=failure_template)

    failures = collect_failures(
        model=model,
        processor=processor,
        device=device,
        dataset_split=ds["validation"],
        prompts=failure_prompts,
        num_samples=num_samples,
        max_failures=10,
    )

    print(f"[FAILURES] Collected {len(failures)} failures")

    Path("results/failures").mkdir(parents=True, exist_ok=True)

    with open("results/failures/failures.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "true_label", "predicted_label"])

        for i, fail in enumerate(failures):
            img_path = f"results/failures/failure_{i}.png"
            fail["image"].save(img_path)

            writer.writerow([
                fail["index"],
                label_names[fail["true"]],
                label_names[fail["pred"]],
            ])

    print("[FAILURES] Saved images and CSV to results/failures/")


if __name__ == "__main__":
    main()
