# Chat with an intelligent assistant in your terminal
from openai import OpenAI
from prompts.contexts import context_document, biography
from prompts.instructs import instructions
from libs.helper import timer_func, num_tokens_from_string

seed_number = 42
welcome_message = "Hello, introduce yourself as Artificial Milad using 2-3 sentences."

# Point to the local server
# client = OpenAI(base_url="http://127.0.0.1:12345/v1", api_key="not-needed")
client = OpenAI(base_url="http://localhost:12345/v1", api_key="not-needed")

history = [
    {"role": "system", "content": f""" Answer technical work questions based on the following:\n {context_document} \n
    answer personal questions based on the following : \n {biography} \n 
    ### Instruction: {instructions} """},
    {"role": "user", "content": "Say: " +  welcome_message + "Answer: "},
]

print(str(num_tokens_from_string(history[0]['content'], "cl100k_base"))+" Tokens!")

while True:
    completion = client.chat.completions.create(
        model="local-model", # this field is currently unused
        stop = ["###", "user:", "assistant:", "instruction:", "###instruction:", "Instruction:", "User:", "Instructions:"],
        messages=history,
        temperature=0.3,
        frequency_penalty=1,
        stream=True,
        seed=seed_number
    )

    new_message = {"role": "assistant", "content": ""}
    
    for chunk in completion:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
            new_message["content"] += chunk.choices[0].delta.content

    history.append(new_message)

    history.append({"role": "user", "content": input("> ")})
    