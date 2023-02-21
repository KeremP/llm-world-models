from typing import List

class Belief:
    embedding: List[float]
    belief: str
    confidence: float

class Evidence:
    fact: str
    source: str