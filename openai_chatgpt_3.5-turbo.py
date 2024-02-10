# Chat with an intelligent assistant in your terminal
from openai import OpenAI
from prompts.contexts import context_document, biography
from prompts.instructs import instructions
from libs.helper import num_tokens_from_string
from settings.open_ai_api import OPEN_AI_SECRET

welcome_message = "Hello, introduce yourself as Artificial Milad using 2-3 sentences."
client = OpenAI(api_key=OPEN_AI_SECRET,)#Read at https://platform.openai.com/docs/api-reference/chat/create

history = [
    {"role": "system", "content": f""" Answer technical work questions based on the following:\n {context_document} \n
    answer personal questions based on the following : \n {biography} \n 
    ### Instruction: {instructions} """},
    {"role": "user", "content": "Say: " +  welcome_message + "Answer: "},
]

print(str(num_tokens_from_string(history[0]['content'], "cl100k_base"))+" Tokens!")

while True:
    completion = client.chat.completions.create(
        model="gpt-4-0125-preview",#"gpt-4",#"gpt-3.5-turbo-0125",
        messages=history,
        stream=True,
    )

    new_message = {"role": "assistant", "content": ""}
    
    for chunk in completion:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
            new_message["content"] += chunk.choices[0].delta.content

    history.append(new_message)

    history.append({"role": "user", "content": input("> ")})
    