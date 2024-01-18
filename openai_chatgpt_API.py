# Chat with an intelligent assistant in your terminal
from openai import OpenAI
from prompts.contexts import context_document, biography
from prompts.instructs import instructions
from libs.helper import num_tokens_from_string
from settings.open_ai_api import OPEN_AI_SECRET

seed_number = 42
welcome_message = "Hello, introduce yourself as Artificial Milad using 2-3 sentences."

client = OpenAI(base_url="https://api.openai.com/v1/chat/gpt-3.5-turbo-instruct/completions", api_key="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",)#Read at #https://platform.openai.com/docs/api-reference/chat/create

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
    