# Chat with an intelligent assistant in your terminal
from openai import OpenAI
from prompts.prompts_resume import Resume

# Point to the local server
client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")


history = [
    {"role": "system", "content": f"You are an intelligent assistant implemented by a Data Scintist named Milad Shirazi, in order to mimic his responses based on his given resume: /n{Resume}"},
    {"role": "user", "content": "Hello, introduce yourself to someone opening this program for the first time. Be concise."},
]


while True:
    completion = client.chat.completions.create(
        model="local-model", # this field is currently unused
        messages=history,
        temperature=0.7,
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