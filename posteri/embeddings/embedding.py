import os
from typing import List
from ..models.document import Evidence

from dotenv import load_dotenv
import openai
from openai.embeddings_utils import get_embedding
load_dotenv()
openai.api_key = os.getenv("OPENAI_SECRET_KEY")


PROMPT = """For each source separated by *, extract three facts. Produce one list for each source. Each list should follow the form:
Source: {source_title}
Facts: {extracted_facts}"""

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

