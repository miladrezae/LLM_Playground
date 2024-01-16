# Chat with an intelligent assistant in your terminal
from openai import OpenAI
from prompts.prompts_resume import context_document, biography
from prompts.instructs import instructions

welcome_message = "Hello, introduce yourself as Artificial Milad using 2-3 sentences."

# Point to the local server
# client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")
client = OpenAI(base_url="http://127.0.0.1:1234/v1", api_key="not-needed")

history = [
    {"role": "system", "content": f""" Answer technical work questions based on the following:\n {context_document} \n
    answer personal questions based on the following : \n {biography} \n 
    ### Instruction: {instructions} """},
    {"role": "user", "content": "Say: " +  welcome_message + "Answer: "},
]


while True:
    completion = client.chat.completions.create(
        model="local-model", # this field is currently unused
        stop = "5",
        messages=history,
        temperature=0.3,
        stream=True,
    )

    new_message = {"role": "assistant", "content": ""}
    
    for chunk in completion:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
            new_message["content"] += chunk.choices[0].delta.content

    history.append(new_message)
    
    # Uncomment to see chat history
    # import json
    # gray_color = "\033[90m"
    # reset_color = "\033[0m"
    # print(f"{gray_color}\n{'-'*20} History dump {'-'*20}\n")
    # print(json.dumps(history, indent=2))
    # print(f"\n{'-'*55}\n{reset_color}")

    print()
    history.append({"role": "user", "content": input("> ")})