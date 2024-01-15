from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
prompt = [
    "[INST] You are an immigration consultant, helping people immigrate to Canada. your task is to answer based on your legal immigration knowledge of Canada. Avoid using links! [/INST] . I'm a data scientist and my employer at Canada is willing to sponsor me for relocation, giving me a chance to have a work permit. Can I bring my spouse with me, even though we are not married? If so, What documents do we need to upload?",
    # "[INST] You are a pissed off person with anger management issues, who is not afraid to use bad words and roast people. your task is to reply based on that mood. Avoid using links! [/INST] . Why are some people allowed to bring their cats to a dog park?"
]
# prompt = "[INST] You are an immigration consultant, helping people immigrate to Canada. your task is to answer based on your legal immigration knowledge of Canada: [/INST] Can somebody who has an closed work permit visa bring their partner even if they are not married? If so, what documentations and proofs are needed to get the visa for the girlfriend/boyfriend?"
# prompt = "[INST] Your task is to help answer the questions of programmers who are writing test codes. Avoid sending links! [/INST] How can I test functionality in Storybook in testing?"
# prompt = "Answer quick and precise! [What is the time difference between Iran and Australia?"

device = "cuda"
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1", device_map="auto", load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", padding_side="left")
tokenizer.pad_token = tokenizer.eos_token  # Most LLMs don't have a pad token by default
model_inputs = tokenizer(
    prompt, return_tensors="pt", padding=True
).to(device)

# generated_ids = model.generate(**model_inputs,max_new_tokens=500,pad_token_id=tokenizer.eos_token_id, repetition_penalty = 100.0, do_sample = True, early_stopping = True, num_beams = 3)
generated_ids = model.generate(**model_inputs,max_new_tokens=150,pad_token_id=tokenizer.eos_token_id, repetition_penalty = 100.0, do_sample = True, early_stopping = True, num_beams = 3)
print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True))
