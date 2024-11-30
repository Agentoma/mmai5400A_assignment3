# Install necessary libraries
!pip install "setfit[absa]" spacy
!spacy download en_core_web_sm

# Ensure compatibility
!pip install tokenizers>=0.20,<0.21
!pip install transformers -U

from setfit import AbsaModel, AbsaTrainer, TrainingArguments
from datasets import load_dataset
from transformers import EarlyStoppingCallback

# Initialize the AbsaModel
model = AbsaModel.from_pretrained(
    "sentence-transformers/all-MiniLM-L6-v2",
    spacy_model="en_core_web_sm",
)

# Load the dataset
dataset = load_dataset("tomaarsen/setfit-absa-semeval-restaurants", split="train")

# Prepare training and evaluation datasets
train_dataset = dataset.select(range(128))
eval_dataset = dataset.select(range(128, 256))

# Set up training arguments
args = TrainingArguments(
    output_dir="models",
    num_train_epochs=5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    evaluation_strategy="steps",
    eval_steps=50,
    save_steps=50,
    load_best_model_at_end=True,
    fp16=True
)

# Initialize and train the AbsaTrainer
trainer = AbsaTrainer(
    model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
)
trainer.train()

# Save the trained model
trainer.save_model("absa_model")