from datasets import load_dataset
import datasets
import torch 
from log import log


def preprocess_function(examples, tokenizer):
    
    """
    Preprocesses the input examples by tokenizing text and labels.

    Args:
        examples (dict): A batch of examples containing 'text' and 'label'.
        tokenizer (Tokenizer): The tokenizer to use.

    Returns:
        dict: Dictionary containing tokenized input IDs and label IDs.
    """
    
    prefix = "sentiment classification: "
    inputs = [prefix + text for text in examples["text"]]

    # Tokenize the inputs
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")

    # Tokenize the labels: 1 → positive, 0 → negative
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            ["positive" if label == 1 else "negative" for label in examples["label"]],
            max_length=2, truncation=True, padding="max_length"
        )
    model_inputs["labels"] = labels["input_ids"]
    
    return model_inputs


def build_dataset(tokenizer):
    
    """
    Loads and prepares the IMDB dataset by tokenizing and splitting into train/test/validation.

    Args:
        tokenizer (Tokenizer): The tokenizer used to tokenize text and labels.

    Returns:
        tuple: Tokenized train, test, and evaluation datasets.
    """
    
    # Load IMDB dataset splits
    dataset = load_dataset("imdb", split=['train', 'test', 'unsupervised'])
    log("Dataset loaded successfully", prefix="DATA")

    # Organize into a DatasetDict and split the test set for validation
    dataset = datasets.DatasetDict({
        "train": dataset[0],
        "test": dataset[1],
        "unsupervised": dataset[2]
    })
    dataset["test"], dataset["validation"] = dataset["test"].train_test_split(test_size=0.6, seed=42).values()
    log("Dataset split into train/test/validation", prefix="DATA")
    
    # Apply the preprocessing function
    tokenized_datasets = dataset.map(
        lambda examples: preprocess_function(examples, tokenizer),
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    log("Tokenization completed", prefix="DATA")
    
    # Extract splits
    train_dataset = tokenized_datasets["train"]
    test_dataset = tokenized_datasets["test"]
    eval_dataset = tokenized_datasets["validation"]
    log("Returning train, test, eval datasets", prefix="DATA")
    
    return train_dataset, test_dataset, eval_dataset
