from datasets import load_dataset

def load_food101():
    ds = load_dataset("food101")
    label_names = ds["train"].features["label"].names
    return ds, label_names