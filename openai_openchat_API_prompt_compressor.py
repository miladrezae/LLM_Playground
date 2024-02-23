# Chat with an intelligent assistant in your terminal
from openai import OpenAI
from prompts.contexts import context_document, biography, extra
from prompts.instructs import instructions
from llmlingua import PromptCompressor
import json
from libs.helper import num_tokens_from_string


llm_lingua = PromptCompressor(model_name="mistralai/Mistral-7B-v0.1")


# Point to the local server
client = OpenAI(base_url="http://127.0.0.1:1234/v1", api_key="not-needed")


question = "Try to reply in a friendly way"
instruction =  f"""### Instruction: {instructions} \n"""
contexts_list = [f""" Answer technical work questions based on the following:\n {context_document} + {extra}\n
Answer personal questions based on the following : {biography}"""]
context_in_one = [contexts_list.extend(instruction)]

history = [
    {"role": "system", "content": f""" Answer technical work questions based on the following:\n {context_document} \n
    answer personal questions based on the following : \n {biography} \n 
    ### Instruction: {instructions} 
    """},
    # {"role": "user", "content": "Hi"},
]
print(str(num_tokens_from_string(history[0]['content'], "cl100k_base"))+" Tokens!")

compressed_prompt = llm_lingua.compress_prompt(
    context=contexts_list,
    instruction=instruction,
    question=question,
    target_token=1000,
    condition_compare=True,
    condition_in_question='after',
    rank_method='longllmlingua',
    use_sentence_level_filter=False,
    context_budget="+500",
    dynamic_context_compression_ratio=0.4, # enable dynamic_context_compression_ratio
    reorder_context="sort"
)
print(json.dumps(compressed_prompt, indent=4))
history = [
    {"role": "system", "content": compressed_prompt["compressed_prompt"]},
]
# print(history)
# while True:
#     completion = client.chat.completions.create(
#         model="local-model", # this field is currently unused
#         stop = ["###", "user:", "assistant:", "instruction:", "###instruction:", "Instruction:", "User:", "Instructions:"],
#         messages=history,
#         temperature=0.3,
#         stream=True,
#     )

#     new_message = {"role": "assistant", "content": ""}
    
#     for chunk in completion:
#         if chunk.choices[0].delta.content:
#             print(chunk.choices[0].delta.content, end="", flush=True)
#             new_message["content"] += chunk.choices[0].delta.content

#     history.append(new_message)
    
#     print(json.dumps(compressed_prompt, indent=4))
#     history.append({"role": "user", "content": input("> ")})

print(compressed_prompt)
with open('prompts/compressed_prompt.txt', 'w', encoding='utf-8') as f:
    f.write(compressed_prompt["compressed_prompt"])

print(str(num_tokens_from_string(compressed_prompt["compressed_prompt"], "cl100k_base"))+" Tokens!")

"""
Conclusion:
This was not to my liking, since I'm calling an API on a local server that's already running a different model(Openchat) in a seperate operating system,
and this compressor requires me to have a seperate model already launched in this environment!
Basically, first it loads a gigantic model on the compressor, then does some process, then connects to my other model on my localhost that's already using my GPU!
I guess this can only be useful if model and compressor are in the same environment? I mean what is the point of a compressor if it needs to initialize a giant model?
Plus it only compressed my data to 70% of first size. 
I will use it once the prompt is final, to reduce size of final prompt.
"""