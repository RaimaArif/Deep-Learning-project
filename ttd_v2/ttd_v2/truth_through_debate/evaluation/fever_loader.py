"""
evaluation/fever_loader.py
Load and sample claims from the FEVER dataset.
"""
from __future__ import annotations
import json
import random
from pathlib import Path

LABEL_MAP = {"SUPPORTS": "TRUE", "REFUTES": "FALSE", "NOT ENOUGH INFO": "NOT ENOUGH INFO"}


def load_fever(path: str, n: int = 100, seed: int = 42,
               balanced: bool = True) -> list[tuple[str, str]]:
    """
    Load n claims from a FEVER .jsonl file.
    Returns list of (claim, label) tuples.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"FEVER file not found: {path}\n"
            "Download from https://fever.ai/dataset/fever.html"
        )
    records = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line.strip())
            label = LABEL_MAP.get(obj.get("label","").upper(), "NOT ENOUGH INFO")
            records.append((obj["claim"], label))

    rng = random.Random(seed)
    if balanced:
        buckets: dict[str, list] = {"TRUE":[], "FALSE":[], "NOT ENOUGH INFO":[]}
        for c, l in records:
            buckets[l].append((c, l))
        per = n // 3
        sample = []
        for items in buckets.values():
            rng.shuffle(items)
            sample.extend(items[:per])
        used = set(id(x) for x in sample)
        rng.shuffle(records)
        for item in records:
            if len(sample) >= n:
                break
            if id(item) not in used:
                sample.append(item)
        return sample[:n]

    rng.shuffle(records)
    return records[:n]


# Built-in 50-claim benchmark (no download needed)
BUILTIN_50 = [
    ("The Great Wall of China is visible from space with the naked eye.", "FALSE"),
    ("Water boils at 100 degrees Celsius at sea level.", "TRUE"),
    ("Humans only use 10% of their brains.", "FALSE"),
    ("The speed of light in a vacuum is approximately 299,792 km/s.", "TRUE"),
    ("Goldfish have a memory span of only three seconds.", "FALSE"),
    ("DNA is a double helix structure.", "TRUE"),
    ("The Earth is approximately 4.5 billion years old.", "TRUE"),
    ("Lightning never strikes the same place twice.", "FALSE"),
    ("Antibiotics are effective against viral infections.", "FALSE"),
    ("The human body has 206 bones in adulthood.", "TRUE"),
    ("Eating carrots improves your night vision significantly.", "FALSE"),
    ("Blood is blue inside the body.", "FALSE"),
    ("Diamonds are made of carbon.", "TRUE"),
    ("The sun is a star.", "TRUE"),
    ("Bats are blind.", "FALSE"),
    ("Humans share about 98% of their DNA with chimpanzees.", "TRUE"),
    ("Vaccines cause autism.", "FALSE"),
    ("The Earth revolves around the Sun.", "TRUE"),
    ("Napoleon Bonaparte was unusually short for his time.", "FALSE"),
    ("The Berlin Wall fell in 1989.", "TRUE"),
    ("Christopher Columbus was the first European to reach the Americas.", "FALSE"),
    ("The first moon landing occurred in 1969.", "TRUE"),
    ("Albert Einstein failed mathematics in school.", "FALSE"),
    ("The Titanic sank on its maiden voyage.", "TRUE"),
    ("World War II ended in 1945.", "TRUE"),
    ("The ancient Egyptians built the pyramids using alien technology.", "FALSE"),
    ("Julius Caesar was assassinated on the Ides of March.", "TRUE"),
    ("The Cold War involved direct military combat between the US and USSR.", "FALSE"),
    ("The printing press was invented by Johannes Gutenberg.", "TRUE"),
    ("Cleopatra was Egyptian by ethnicity.", "FALSE"),
    ("Australia is both a country and a continent.", "TRUE"),
    ("The Amazon River is the longest river in the world.", "FALSE"),
    ("Mount Everest is the tallest mountain on Earth measured from sea level.", "TRUE"),
    ("The Sahara is the world's largest desert.", "FALSE"),
    ("Russia is the largest country in the world by area.", "TRUE"),
    ("The capital of Australia is Sydney.", "FALSE"),
    ("Africa is the world's largest continent.", "FALSE"),
    ("The Pacific Ocean is the largest ocean on Earth.", "TRUE"),
    ("Iceland is covered mostly in ice and Greenland is covered in grass.", "FALSE"),
    ("The Nile is located in Africa.", "TRUE"),
    ("The first iPhone was released in 2007.", "TRUE"),
    ("The internet was invented by Tim Berners-Lee.", "FALSE"),
    ("Python is a compiled programming language.", "FALSE"),
    ("Google was founded in 1998.", "TRUE"),
    ("The first commercial email was sent in 1971.", "TRUE"),
    ("Facebook was founded by Mark Zuckerberg alone.", "FALSE"),
    ("The first computer bug was an actual insect.", "TRUE"),
    ("WiFi stands for Wireless Fidelity.", "FALSE"),
    ("The first video game ever made was Pong.", "FALSE"),
    ("Binary code uses only 0s and 1s.", "TRUE"),
]


def load_builtin(n: int = 50, seed: int = 42) -> list[tuple[str, str]]:
    data = BUILTIN_50[:n]
    random.Random(seed).shuffle(data)
    return data
