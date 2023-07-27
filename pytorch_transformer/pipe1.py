from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# simplest useage
classifier = pipeline("sentiment-analysis")

res = classifier("I've been waiting for a HuggingFace course my whole life.")

print(res)

# more complex useage, specifying model and tokenizer
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
res = classifier("I've been waiting for a HuggingFace course my whole life.")
print(res)

# using tokenizer
sequence = "Using a transformer network is simple"
res = tokenizer(sequence)
print(res)
tokens = tokenizer.tokenize(sequence)
print(tokens)
ids = tokenizer.convert_tokens_to_ids(tokens)
print(ids)
decoded_string = tokenizer.decode(ids)
print(decoded_string)

# useful for fine-tuning
X_train = ["I've been waiting for a HuggingFace course my whole life.", "Python is great!"]
res = classifier(X_train)
print(res)

batch = tokenizer(X_train, padding=True, truncation=True, max_length=512, return_tensors="pt")
print(batch)

with torch.no_grad():
    outputs = model(**batch)
    print(outputs)
    predictions = F.softmax(outputs.logits, dim=1)
    print(predictions)
    labels = torch.argmax(predictions, dim=1)
    print(labels)

# save and load model
save_directory = "saved"
tokenizer.save_pretrained(save_directory)
model.save_pretrained(save_directory)

tok = AutoTokenizer.from_pretrained(save_directory)
mod = AutoModelForSequenceClassification.from_pretrained(save_directory)


