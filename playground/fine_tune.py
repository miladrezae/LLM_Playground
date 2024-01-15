
from datasets import load_dataset
import evaluate
from tqdm.auto import tqdm
import torch
from torch.optim import AdamW
from transformers import get_scheduler
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from torchsummary import summary

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
dataset = load_dataset("yelp_review_full")
# print(dataset)
# print(dataset["test"][100])

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
"""Now we remove texts and rename label to labels and make sure dataset outputs torch tensors:"""
tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

"""Now we load smaller data into dataloaders"""
small_train_dataset = tokenized_datasets["train"]#.shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"]#.shuffle(seed=42).select(range(1000))

train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=30)
eval_dataloader = DataLoader(small_eval_dataset, batch_size=30)

"""Load pretrained model:"""
model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)
print(f"No. of parameters:{model.num_parameters()}")
# print(repr(model))
model.to(device)

"""Evaluation mode for the pretrained model"""
model.eval()
metric_default = evaluate.load("accuracy")
progress_bar_default = tqdm(range(len(eval_dataloader)))
with torch.no_grad():
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        logits = outputs.logits
        progress_bar_default.update(1)
    predictions = torch.argmax(logits, dim=-1)
metric_default.add_batch(predictions=predictions, references=batch["labels"])

print(metric_default.compute())

print("______")

"""Optimizer and lr scheduler"""
optimizer = AdamW(model.parameters(), lr=5e-5)
num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

"""Retrain"""
progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)


"""Evaluate"""
metric = evaluate.load("accuracy")
model.eval()
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

print(metric.compute())
torch.save(model, "model.pth")
#torch.load('model.pth')