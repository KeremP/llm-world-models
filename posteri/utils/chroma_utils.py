from typing import Optional, List, Callable
import uuid
import chromadb
from chromadb.api.models import Collection
from ..models.document import Belief

def create_collection(client: Optional[chromadb.Client] = None) -> Collection:
    collection_name = uuid.uuid4().hex
    if client is None:
        client = chromadb.Client()
    collection = client.create_collection(name=collection_name)
    return collection

def build_docstore(docs: List[Belief]):
    collection = create_collection()
    collection.add(
        embeddings=[doc.embedding for doc in docs],
        documents=[doc.belief for doc in docs],
        metadatas=[{"confidence":doc.confidence} for doc in docs],
        ids=[uuid.uuid4().hex for _ in docs]
    )
    return collection

def search_collection(query: str, collection: Collection, embedding_func: Callable, num_results: int, **kwargs):
    embedded_q = embedding_func(query, **kwargs)
    results = collection.query(
        query_embeddings=[embedded_q],
        n_results = num_results
    )
    return results