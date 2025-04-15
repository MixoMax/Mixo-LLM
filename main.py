from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import GPT2Config, GPT2LMHeadModel
from transformers import TrainingArguments, Trainer, pipeline
import time

print("Starting script...")
start_time = time.time()

# Load dataset
print("Loading dataset...")
dataset_start = time.time()
dataset = load_dataset("text", data_files={"train": "data.txt"})
print(f"Dataset loaded in {time.time() - dataset_start:.2f} seconds")

# Load tokenizer
print("Loading tokenizer...")
tokenizer_start = time.time()
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
print(f"Tokenizer loaded in {time.time() - tokenizer_start:.2f} seconds")

def tokenize_function(examples):
    tokenized = tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)
    tokenized['labels'] = tokenized['input_ids'].copy()
    return tokenized

# Tokenize datasets
print("Tokenizing datasets...")
tokenize_start = time.time()
tokenized_datasets = dataset.map(tokenize_function, batched=True)
print(f"Datasets tokenized in {time.time() - tokenize_start:.2f} seconds")

n_samples = 10_000
# only train on 10k samples for now
tokenized_datasets["train"] = tokenized_datasets["train"].select(range(n_samples))

# Initialize model
print("Initializing model...")
model_start = time.time()
config = GPT2Config(
    vocab_size=tokenizer.vocab_size,
    n_positions=512,
    n_ctx=512,
    n_embd=256,
    n_layer=4,
    n_head=4,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)
model = GPT2LMHeadModel(config)
print(f"Model initialized in {time.time() - model_start:.2f} seconds")

training_args = TrainingArguments(
    output_dir='./results',
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=16,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir='./logs',
    logging_steps=500,
    prediction_loss_only=True,
)

# Training
print("Starting training...")
training_start = time.time()
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
)

trainer.train()
print(f"Training completed in {time.time() - training_start:.2f} seconds")
print(f"Total execution time: {time.time() - start_time:.2f} seconds")


# save model
model.save_pretrained("model")
tokenizer.save_pretrained("model")
