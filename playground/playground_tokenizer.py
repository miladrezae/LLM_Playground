from transformers import AutoTokenizer

input = ["Do not meddle in the affairs of wizards! or should you?","Don't think he knows about second breakfast, Pip.","What about elevensies?"]
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

encoded_input = tokenizer(input)
encoded_input_padded = tokenizer(input, padding=True)
encoded_input_truncated = tokenizer(input, truncation=True)
encoded_input_tensor = tokenizer(input,truncation=True, padding=True,return_tensors="pt") #pt or tf

decoded_output = tokenizer.decode(encoded_input["input_ids"][-1])
print(encoded_input)
print(f"Padded: {encoded_input_padded}")
print(f"Padded: {encoded_input_truncated}")
print(decoded_output)
print(f"Tensor: {encoded_input_tensor}")