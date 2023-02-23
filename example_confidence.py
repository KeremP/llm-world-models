#!/usr/bin/env python
import os
from dotenv import load_dotenv
load_dotenv()
from posteri.embeddings.embedding import get_confidence, calculate_posterior
from posteri.utils.chroma_utils import create_collection
from posteri.utils.chroma_utils import create_collection, build_docstore, search_collection
from posteri.models.document import Belief
import openai
from openai.embeddings_utils import get_embedding

openai.api_key = os.getenv("OPENAI_API_KEY")

CONTEXT = [
    ("Kerem is from Turkey", 1.0),
    ("Kerem likes to cook", .90),
]

def create_beliefs(context=CONTEXT):
    out = []
    for c in context:
        emb = get_embedding(c[0], engine="text-embedding-ada-002")
        out.append(
            Belief(embedding=emb, belief=c[0], confidence=c[1])
        )
    return out

BELIEFS = create_beliefs()
docstore = build_docstore(BELIEFS)

confidence = get_confidence("Kerem likes spicy food", docstore, embedding_func=get_embedding, num_results=2, engine="text-embedding-ada-002")
print(confidence)

posterior = calculate_posterior(["Kerem likes spicy food"], "Kerem ordered spicy gumbo", confidence)
print(posterior)

