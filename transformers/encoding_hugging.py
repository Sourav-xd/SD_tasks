from transformers import BertTokenizer, BertForMaskedLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from torch.utils.data import Dataset
import torch

# 1. Prepare Dataset
class SimpleTextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len=128):
        self.encodings = tokenizer(texts, truncation=True, padding='max_length', max_length=max_len)
        
    def __getitem__(self, idx):
        # Transformers requires inputs as a list of dicts/objects
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item
        
    def __len__(self):
        return len(self.encodings['input_ids'])

# 2. Setup Model and Tokenizer
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForMaskedLM.from_pretrained(model_name)

# Sample Data (In reality, this would be a large corpus)
texts =
dataset = SimpleTextDataset(texts, tokenizer)

# 3. Data Collator: The Magic Component
# This automatically handles the random masking strategy (80/10/10 rule).
# We don't need to manually replace tokens with; this class does it on the fly.
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, 
    mlm=True, 
    mlm_probability=0.15  # 15% of tokens masked
)

# 4. Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=10,
    logging_steps=10
)

# 5. Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

# 6. Train
# trainer.train() 

# 7. Inference: Predicting a Mask
# To prove proficiency, we manually encode a masked sentence and decode the prediction.
input_text = "The capital of France is."
inputs = tokenizer(input_text, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

# Retrieve index of
mask_token_index = (inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)

# Get the token with the highest probability at that index
predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
decoded_word = tokenizer.decode(predicted_token_id)

print(f"Sentence: {input_text}")
print(f"Predicted: {decoded_word}") 
# Expected output: paris