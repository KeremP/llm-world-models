import os
import numpy as np
from typing import List
from ..models.document import Evidence, Belief
from ..utils.chroma_utils import search_collection
from ..prompts.prompts import BELIEF_PROMPT, NEGATE_PROMPT, POSTERIOR_PROMPT

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

def get_proba(results):
    log_probs = results.choices[0]['logprobs']['token_logprobs']
    tokens: list = results.choices[0]['logprobs']['tokens']
    token = results.choices[0]['text'].strip("\n")
    idx = tokens.index(token)
    log_prob = log_probs[idx]
    proba = np.e**log_prob
    return proba, token

def get_confidence(belief: str, collection, **kwargs) -> float:
    context_results = search_collection(belief, collection, **kwargs)
    docs = context_results['documents'][0]
    context = "\n".join(["- "+d for d in docs])

    prompt = PromptTemplate(
        input_variables=['context','belief'],
        template=BELIEF_PROMPT
    )
    
    prompt_formatted = prompt.format(context=context, belief=belief)
    results = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt_formatted,
        temperature=0,
        logprobs=5,
    )
    print(prompt_formatted)
    confidence, token = get_proba(results)
    print(token)
    if "no" in token.lower():
        confidence = 1 - confidence
    if confidence > .90: confidence = .90
    return confidence


def calculate_posterior(beliefs: List[str], evidence: str, confidence: float) -> float:
    positive_prompt = PromptTemplate(
        input_variables=['beliefs','evidence'],
        template=POSTERIOR_PROMPT
    )

    negation_prompt = PromptTemplate(
        input_variables=['beliefs'],
        template=NEGATE_PROMPT
    )
    negate_formatted = negation_prompt.format(beliefs=". ".join(beliefs))
    negate_results = openai.Completion.create(
            model="text-davinci-003",
            prompt=negate_formatted,
            temperature=0,
        )
    negated_beliefs = negate_results.choices[0].text.strip("\n")
    print(negated_beliefs)

    pos_prompt_1 = positive_prompt.format(beliefs=negated_beliefs, evidence=evidence)
    evidence_not_belief = openai.Completion.create(
            model="text-davinci-003",
            prompt=pos_prompt_1,
            temperature=0,
            logprobs=5
        )

    pos_prompt_2 = positive_prompt.format(beliefs=". ".join(beliefs), evidence=evidence)
    evidence_belief = openai.Completion.create(
            model="text-davinci-003",
            prompt=pos_prompt_2,
            temperature=0,
            logprobs=5
        )
    
    prob_evidence_not_belief, _ = get_proba(evidence_not_belief)
    prob_evidence_belief, _ = get_proba(evidence_belief)

    not_confidence = 1 - confidence

    prob_evidence = (confidence * prob_evidence_belief) + (not_confidence * prob_evidence_not_belief)

    posterior = max(confidence * prob_evidence_belief / prob_evidence, .95)

    return posterior





    
    

