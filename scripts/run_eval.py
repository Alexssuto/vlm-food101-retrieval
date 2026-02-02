from src.dataset import load_food101
from src.model import load_clip
from src.evaluate import build_prompts, evaluate_zero_shot

def main():
    ds, label_names = load_food101()
    model, processor, device = load_clip()
    prompts = build_prompts(label_names)

    top1, top5 = evaluate_zero_shot(
        model=model,
        processor=processor,
        device=device,
        dataset_split=ds["validation"],
        prompts=prompts,
        num_samples=500
    )

    print(f"Device: {device}")
    print(f"Top-1: {top1:.3f} | Top-5: {top5:.3f}")

if __name__ == "__main__":
    main()
