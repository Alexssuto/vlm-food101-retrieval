import torch
from tqdm import tqdm
from pathlib import Path


def build_prompts(label_names, template="a photo of {label}"):
    return [template.format(label=name.replace("_", " ")) for name in label_names]


@torch.no_grad()
def evaluate_zero_shot_batched(
    model,
    processor,
    device,
    dataset_split,
    prompts,
    num_samples=2000,
    batch_size=32,
):
    correct_top1 = 0
    correct_top5 = 0

    text_inputs = processor(text=prompts, return_tensors="pt", padding=True).to(device)
    text_out = model.get_text_features(**text_inputs)
    text_features = text_out.pooler_output if hasattr(text_out, "pooler_output") else text_out
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    subset = dataset_split.select(range(num_samples))

    images, labels = [], []

    for ex in tqdm(subset, desc="Evaluating"):
        images.append(ex["image"])
        labels.append(ex["label"])

        if len(images) == batch_size:
            image_inputs = processor(images=images, return_tensors="pt").to(device)
            img_out = model.get_image_features(**image_inputs)
            image_features = img_out.pooler_output if hasattr(img_out, "pooler_output") else img_out
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            logits = image_features @ text_features.T
            top5 = logits.topk(5, dim=1).indices

            true = torch.tensor(labels, device=device)
            correct_top1 += (top5[:, 0] == true).sum().item()
            correct_top5 += (top5 == true.unsqueeze(1)).any(dim=1).sum().item()

            images, labels = [], []

    if images:
        image_inputs = processor(images=images, return_tensors="pt").to(device)
        img_out = model.get_image_features(**image_inputs)
        image_features = img_out.pooler_output if hasattr(img_out, "pooler_output") else img_out
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        logits = image_features @ text_features.T
        top5 = logits.topk(5, dim=1).indices

        true = torch.tensor(labels, device=device)
        correct_top1 += (top5[:, 0] == true).sum().item()
        correct_top5 += (top5 == true.unsqueeze(1)).any(dim=1).sum().item()

    return correct_top1 / num_samples, correct_top5 / num_samples


@torch.no_grad()
def collect_failures(
    model,
    processor,
    device,
    dataset_split,
    prompts,
    num_samples=2000,
    max_failures=10,
):
    """
    Collect misclassified examples (Top-1 failures).
    """
    Path("results/failures").mkdir(parents=True, exist_ok=True)

    text_inputs = processor(text=prompts, return_tensors="pt", padding=True).to(device)
    text_out = model.get_text_features(**text_inputs)
    text_features = text_out.pooler_output if hasattr(text_out, "pooler_output") else text_out
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    subset = dataset_split.select(range(num_samples))
    failures = []

    for idx, ex in enumerate(subset):
        image = ex["image"]
        true_label = ex["label"]

        image_inputs = processor(images=image, return_tensors="pt").to(device)
        img_out = model.get_image_features(**image_inputs)
        image_features = img_out.pooler_output if hasattr(img_out, "pooler_output") else img_out
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        logits = image_features @ text_features.T
        pred_label = logits.argmax(dim=1).item()

        if pred_label != true_label:
            failures.append({
                "index": idx,
                "true": true_label,
                "pred": pred_label,
                "image": image,
            })

        if len(failures) >= max_failures:
            break

    return failures