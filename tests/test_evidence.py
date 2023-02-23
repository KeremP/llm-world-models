#!/usr/bin/env python
import os
import openai
from openai.embeddings_utils import get_embedding
from dotenv import load_dotenv
load_dotenv()

from posteri.embeddings import embedding
from posteri.models.document import Belief
from posteri.utils.chroma_utils import create_collection, build_docstore, search_collection
import pytest

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
# TODO: cache emebedding requests so that tests do not incur costs
class TestEvidence(object):

    def test_create_collection(self):
        collection = create_collection()
        assert collection.name
    
    def test_create_docstore(self):
        docstore = build_docstore(BELIEFS)
        assert docstore.count() == 2

    def test_search_index(self):
        docstore = build_docstore(BELIEFS)
        q = "Kerem likes spicy food"
        results = search_collection(q, docstore, get_embedding, 2, engine="text-embedding-ada-002")
        docs = results['documents'][0]
        metadatas = results['metadatas'][0]
        assert len(docs) == 2
        assert len(metadatas) == 2
