"""
data/sample_claims.py

A small curated set of fact-checking claims for rapid development/testing
without needing to download the full FEVER dataset.

Labels: "SUPPORTS" (TRUE) or "REFUTES" (FALSE)
"""

SAMPLE_CLAIMS: list[tuple[str, str]] = [
    # ── Science & Nature ──────────────────────────────────────────────────
    ("The Great Wall of China is visible from space with the naked eye.", "REFUTES"),
    ("Water boils at 100 degrees Celsius at sea level.", "SUPPORTS"),
    ("Humans only use 10% of their brains.", "REFUTES"),
    ("The speed of light in a vacuum is approximately 299,792 km/s.", "SUPPORTS"),
    ("Goldfish have a memory span of only three seconds.", "REFUTES"),
    ("DNA is a double helix structure.", "SUPPORTS"),
    ("The Earth is approximately 4.5 billion years old.", "SUPPORTS"),
    ("Lightning never strikes the same place twice.", "REFUTES"),

    # ── History ──────────────────────────────────────────────────────────
    ("Napoleon Bonaparte was unusually short for his time.", "REFUTES"),
    ("The Berlin Wall fell in 1989.", "SUPPORTS"),
    ("Christopher Columbus was the first European to reach the Americas.", "REFUTES"),
    ("The first moon landing occurred in 1969.", "SUPPORTS"),
    ("Albert Einstein failed mathematics in school.", "REFUTES"),
    ("The Titanic sank on its maiden voyage.", "SUPPORTS"),

    # ── Geography ────────────────────────────────────────────────────────
    ("Australia is both a country and a continent.", "SUPPORTS"),
    ("The Amazon River is the longest river in the world.", "REFUTES"),
    ("Mount Everest is the tallest mountain on Earth measured from sea level.", "SUPPORTS"),
    ("The Sahara is the world's largest desert.", "REFUTES"),  # Antarctica is larger

    # ── Technology ───────────────────────────────────────────────────────
    ("The first iPhone was released in 2007.", "SUPPORTS"),
    ("The internet was invented by Tim Berners-Lee.", "REFUTES"),  # He invented WWW
    ("Python is a compiled programming language.", "REFUTES"),
    ("Google was founded in 1998.", "SUPPORTS"),

    # ── Biology & Medicine ───────────────────────────────────────────────
    ("Antibiotics are effective against viral infections.", "REFUTES"),
    ("The human body has 206 bones in adulthood.", "SUPPORTS"),
    ("Eating carrots improves your night vision.", "REFUTES"),
    ("Blood is blue inside the body.", "REFUTES"),
]


def load_sample(n: int = None, seed: int = 42) -> list[tuple[str, str]]:
    """
    Return a sample of built-in claims.

    Args:
        n:    Number of claims to return (None = all).
        seed: Random seed for shuffling.

    Returns:
        List of (claim, label) tuples.
    """
    import random
    rng = random.Random(seed)
    claims = list(SAMPLE_CLAIMS)
    rng.shuffle(claims)
    if n is not None:
        claims = claims[:n]
    print(f"[Sample] Loaded {len(claims)} built-in claims.")
    return claims
