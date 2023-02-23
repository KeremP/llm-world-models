import os
import numpy as np
from typing import List
from ..models.document import Evidence, Belief
from ..utils.chroma_utils import search_collection
from ..prompts.prompts import BELIEF_PROMPT

from dotenv import load_dotenv
import openai
from openai.embeddings_utils import get_embedding
from langchain import OpenAI, PromptTemplate
from langchain.chains.llm import LLMChain

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


PROMPT = """For each source separated by *, extract three facts. Produce one list for each source. Do not reference other sources in the list. Each list should be separated with <[]>:"""

def extract_evidence(chunks: List[str]) -> List[Evidence]:
    """
    Extract evidence/facts from chunks of text using a call to the LLM
    
    Returns a list of `Evidence` objects with each fact and their source
    """
    chunks = ["*"+chunk for chunk in chunks]
    prompt = PROMPT + "\n\n" + "\n".join(chunks) + "\n"

    results = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        max_tokens=500,
        temperature=0
    )
    print(prompt)
    print("\n")
    print(results)
    results = results.choices[0].text
    facts = results.split("\n")
    evidence = []
    for i,fact in enumerate(facts):
        # TODO: should ensure results are in order of context
        parsed_facts = fact.split("\n")[2:]
        evidence+=[Evidence(fact=f[2:].strip(), source=chunks[i].strip("*")) for f in parsed_facts]
    return evidence


def create_belief(belief: str, confidence: float) -> Belief:
    embedding = get_embedding(belief, engine="text-embedding-ada-002")
    obj = Belief(embedding=embedding, belief=belief, confidence=confidence)
    return obj

def get_confidence(belief: str, collection, **kwargs) -> float:
    context_results = search_collection(belief, collection, **kwargs)
    docs = context_results['documents'][0]
    context = "\n".join(["- "+d for d in docs]) #TODO: parse results

    prompt = PromptTemplate(
        input_variables=['context','belief'],
        template=BELIEF_PROMPT
    )
    
    prompt_formatted = prompt.format(context=context, belief=belief)
    results = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt_formatted,
        temperature=0,
        logprobs=5
    )
    log_probs = results.choices[0]['logprobs']['token_logprobs']
    tokens: list = results.choices[0]['logprobs']['tokens']
    token = results.choices[0]['text'].strip("\n")
    idx = tokens.index(token)
    log_prob = log_probs[idx]
    confidence = np.e**log_prob
    if confidence > .90: confidence = .90
    return confidence


    
    

