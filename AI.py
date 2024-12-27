import os
import re
import torch
from datasets import Dataset
from transformers import BartForConditionalGeneration, BartTokenizer, Trainer, TrainingArguments


# Utility to clean text
def clean_text(text):
    """Cleans garbled text by removing non-alphanumeric characters and excessive spaces."""
    cleaned = re.sub(r"[^a-zA-Z0-9.,!?;:'\"()\s]", "", text)  # Keep essential punctuation
    cleaned = re.sub(r"\s+", " ", cleaned).strip()  # Replace multiple spaces with single space
    return cleaned


# Function to generate summaries for missing outputs
def generate_summary(text, model, tokenizer):
    """Generates a summary using a pretrained BART model."""
    inputs = tokenizer(text, max_length=1024, truncation=True, return_tensors="pt")
    summary_ids = model.generate(
        inputs.input_ids, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


# Utility to load paired text or generate outputs
def load_text_files(input_dir, output_dir, model, tokenizer):
    """Load, clean, and pair input-output text; generate outputs if missing."""
    inputs, outputs = [], []

    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists

    for input_file in sorted(os.listdir(input_dir)):
        try:
            input_path = os.path.join(input_dir, input_file)
            output_path = os.path.join(output_dir, input_file)

            # Load and clean input text
            with open(input_path, "r", encoding="utf-8") as f_in:
                raw_input = f_in.read().strip()
                cleaned_input = clean_text(raw_input)
                if not cleaned_input:
                    print(f"Skipping empty input file: {input_file}")
                    continue
                inputs.append(cleaned_input)

            # Load or generate output text
            if os.path.exists(output_path):
                with open(output_path, "r", encoding="utf-8") as f_out:
                    raw_output = f_out.read().strip()
                    cleaned_output = clean_text(raw_output)
                    outputs.append(cleaned_output)
            else:
                print(f"Generating output for: {input_file}")
                generated_output = generate_summary(cleaned_input, model, tokenizer)
                outputs.append(generated_output)
                # Save the generated output
                with open(output_path, "w", encoding="utf-8") as f_out:
                    f_out.write(generated_output)

            print(f"Loaded pair - Input: '{cleaned_input[:50]}...', Output: '{outputs[-1][:50]}...'")

        except Exception as e:
            print(f"Skipping file: {input_file} due to error {e}")

    return inputs, outputs


# Tokenization pipeline
class TokenizerHandler:
    """Wrapper for tokenizer to preprocess text for BART."""

    def __init__(self, model_name):
        self.tokenizer = BartTokenizer.from_pretrained(model_name)

    def preprocess(self, example):
        """Tokenize both input and target data."""
        source = self.tokenizer(
            example["input"], max_length=512, truncation=True, padding="max_length"
        )
        target = self.tokenizer(
            example["output"], max_length=512, truncation=True, padding="max_length"
        )
        source["labels"] = target["input_ids"]
        return source


# Split dataset into training and validation sets
def split_dataset(dataset, test_size=0.1):
    """Split the dataset into training and validation subsets."""
    adjusted_test_size = min(test_size, len(dataset) - 1) / len(dataset)  # Avoid empty splits
    print(f"Using test_size: {adjusted_test_size:.2f}")
    split_data = dataset.train_test_split(test_size=adjusted_test_size)
    return split_data["train"], split_data["test"]


# Set up training arguments
def get_training_args():
    """Define training arguments."""
    return TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=1,  # Use the smallest batch size
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,  # Accumulate gradients over 8 steps
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=500,
        load_best_model_at_end=True,
        save_total_limit=2,
    )





# Main orchestration function
def main():
    # Directory configuration
    input_dir = "./input"
    output_dir = "./output"
    model_name = "facebook/bart-large-cnn"  # Pretrained model for summarization

    # Load pretrained model and tokenizer for output generation
    print("Loading pretrained BART model...")
    model = BartForConditionalGeneration.from_pretrained(model_name)
    tokenizer = BartTokenizer.from_pretrained(model_name)

    # Load data
    print("Loading and cleaning data...")
    inputs, outputs = load_text_files(input_dir, output_dir, model, tokenizer)
    if len(inputs) == 0 or len(outputs) == 0:
        raise ValueError("No valid input-output pairs found. Check your data files.")

    # Prepare dataset
    print(f"Total pairs loaded: {len(inputs)}")
    data = {"input": inputs, "output": outputs}
    dataset = Dataset.from_dict(data)

    # Tokenize dataset
    print("Tokenizing data...")
    tokenizer_handler = TokenizerHandler(model_name)
    tokenized_dataset = dataset.map(tokenizer_handler.preprocess, batched=True, remove_columns=["input", "output"])

    # Split dataset
    print("Splitting dataset into train/validation...")
    train_dataset, val_dataset = split_dataset(tokenized_dataset)

    # Initialize model for fine-tuning
    print("Loading model for fine-tuning...")
    model = BartForConditionalGeneration.from_pretrained(model_name)

    # Training arguments
    print("Setting up training...")
    training_args = get_training_args()
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # Train the model
    print("Starting training...")
    trainer.train()
    print("Training completed successfully!")


if __name__ == "__main__":
    main()
