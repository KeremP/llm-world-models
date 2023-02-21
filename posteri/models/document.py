from typing import List
from pydantic import BaseModel

class Belief(BaseModel):
    embedding: List[float]
    belief: str
    confidence: float

class Evidence(BaseModel):
    fact: str
    source: str