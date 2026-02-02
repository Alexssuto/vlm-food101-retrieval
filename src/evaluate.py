import torch
from tqdm import tqdm

def build_prompts(label_names):
    return [f"a photo of {name.replace('_', ' ')}" for name in label_names]

@torch.no_grad()
def evaluate_zero_shot(model, processor, device, dataset_split, prompts, num_samples=50):
    correct_top1 = 0
    correct_top5 = 0

    text_inputs = processor(text=prompts, return_tensors="pt", padding=True).to(device)
    subset = dataset_split.select(range(num_samples))

    for ex in tqdm(subset, desc="Evaluating"):
        image = ex["image"]
        true_label = ex["label"]

        image_inputs = processor(images=image, return_tensors="pt").to(device)
        outputs = model(**image_inputs, **text_inputs)

        probs = outputs.logits_per_image.softmax(dim=1)
        top5 = probs.topk(5, dim=1).indices[0].tolist()

        if top5[0] == true_label:
            correct_top1 += 1
        if true_label in top5:
            correct_top5 += 1

    return correct_top1 / num_samples, correct_top5 / num_samples
