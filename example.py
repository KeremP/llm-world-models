#!/usr/bin/env python
from posteri.agent.agent import init_agent, generate_beliefs

if __name__ == "__main__":
    print("INITIALZING AGENT...")    
    agent = init_agent()
    history = []

    while True:
        human_input = input(">> ")
        if human_input == "END":
            break
        pred = agent.predict(human_input=human_input)
        history.append(human_input)
        print(pred)
    
    beliefs = generate_beliefs(history, 5)
    print(beliefs)
