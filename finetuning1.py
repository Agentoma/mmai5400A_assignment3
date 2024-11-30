import pandas as pd
from datasets import Dataset
from setfit import AbsaModel, AbsaTrainer
from transformers import TrainingArguments

# Load your custom data
data = pd.read_csv(
    "reviews.csv",
    delimiter=",",
    on_bad_lines="skip",
    quoting=3
)

# Prepare the DataFrame
train_data = pd.DataFrame({
    'text': data['Review'],
    'span': data['Name'],
    'polarity': data['RatingValue']
})

# Map polarity to ordinal values
polarity_to_ordinal = {
    1: 0,
    2: 1,
    3: 2,
    4: 3,
    5: 4
}
train_data['ordinal'] = train_data['polarity'].map(polarity_to_ordinal)

# Convert DataFrame to Dataset
train_dataset = Dataset.from_pandas(train_data)
eval_dataset = train_dataset  # Or use a separate evaluation dataset

# Initialize the model
model = AbsaModel.from_pretrained(
    "sentence-transformers/all-MiniLM-L6-v2",
    spacy_model="en_core_web_sm"
)

# Define training arguments
args = TrainingArguments(
    output_dir="models",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    evaluation_strategy="steps",
    eval_steps=50,
    save_steps=50,
    load_best_model_at_end=True,
    fp16=True
)

# Initialize the trainer
trainer = AbsaTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

# Train the model
trainer.train()

# Evaluate the model
metrics = trainer.evaluate()
print(metrics)