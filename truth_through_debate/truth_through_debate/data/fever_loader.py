"""
data/fever_loader.py
FEVER dataset loader using mteb/fever — parquet format, no loading script needed.
185,445 claims labeled SUPPORTS / REFUTES / NOT ENOUGH INFO.
"""
from datasets import load_dataset


def load_fever(
    split: str = "test",
    n: int = 100,
    exclude_nei: bool = True,
    seed: int = 42,
) -> list[tuple[str, str]]:
    """
    Load claims from FEVER via mteb/fever (parquet, no trust_remote_code needed).

    Args:
        split:       "train" or "test"
        n:           Max number of claims to return
        exclude_nei: Skip NOT ENOUGH INFO claims
        seed:        Random seed for shuffling

    Returns:
        List of (claim, label) tuples. label = "SUPPORTS" or "REFUTES"
    """
    ds = load_dataset("mteb/fever", split=split)
    ds = ds.shuffle(seed=seed)

    claims = []
    for example in ds:
        raw_label = example.get("label", 2)
        if isinstance(raw_label, int):
            label = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"][min(raw_label, 2)]
        else:
            label = str(raw_label).upper()

        if exclude_nei and label == "NOT ENOUGH INFO":
            continue

        claim = example.get("text") or example.get("claim", "")
        if not claim:
            continue

        claims.append((claim, label))
        if len(claims) >= n:
            break

    print(f"[FEVER] Loaded {len(claims)} claims from mteb/fever ({split} split).")
    return claims


def label_distribution(claims: list[tuple[str, str]]) -> dict:
    dist = {}
    for _, label in claims:
        dist[label] = dist.get(label, 0) + 1
    return dist
