import os
import openai
from dotenv import load_dotenv
from langchain import OpenAI, ConversationChain, PromptTemplate
from langchain.chains.conversation.memory import ConversationalBufferWindowMemory
from langchain.chains.llm import LLMChain

from ..prompts.prompts import TEMPLATE, EXTRACT_FACTS_PROMPT

from typing import Union, List

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")



def init_agent() -> LLMChain:
    PROMPT = PromptTemplate(
        input_variables=["history", "human_input"],
        template=TEMPLATE
    )

    agent = LLMChain(
        llm=OpenAI(temperature=0.7),
        prompt=PROMPT,
        verbose=True,
        memory=ConversationalBufferWindowMemory(k=5)
    )
    return agent

def generate_beliefs(conversation: List[str], num_facts: int):
    # TODO: validate length of conversation
    conversation = "\n".join(conversation)
    EXTRACT_PROMPT = PromptTemplate(
        input_variables=["num_facts","conversation"],
        template=EXTRACT_FACTS_PROMPT
    )

    extract_agent = LLMChain(
        llm=OpenAI(temperature=0),
        prompt=EXTRACT_PROMPT,
        verbose=True
    )
    beliefs = extract_agent.predict(num_facts=num_facts, conversation=conversation)
    # TODO: extract beliefs
    return beliefs

    
