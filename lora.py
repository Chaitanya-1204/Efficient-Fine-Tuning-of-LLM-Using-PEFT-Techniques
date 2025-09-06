from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments,DataCollatorForSeq2Seq, T5ForConditionalGeneration
import torch
from peft import get_peft_model, LoraConfig, TaskType

from utils import compute_metrics
from log import log


def lora(tokenizer, train_dataset, test_dataset, eval_dataset, device):
    
    """
    Fine-tunes a T5 model using LoRA with multiple rank settings and evaluates it.

    Args:
        tokenizer (Tokenizer): Tokenizer for preprocessing.
        train_dataset (Dataset): Training dataset.
        test_dataset (Dataset): Test dataset.
        eval_dataset (Dataset): Validation dataset.
        device (torch.device): Device to run the model on.
    """
    
    lora_rank = [4, 8, 16, 32, 64, 128]

    # Set up data collator for seq2seq tasks
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=None)

    for r in lora_rank:
        log(f"\n\n\n=== Training with LoRA Rank = {r} ===\n\n\n")

        # Define LoRA configuration for current rank
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            inference_mode=False,
            r=r,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none"
        )

        # Load base model and apply LoRA
        model_name = "t5-small"
        model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        # Update data collator with the model
        data_collator.model = model

        # Configure training arguments
        training_args = Seq2SeqTrainingArguments(
            output_dir=f"t5_lora_{r}_imdb",
            eval_strategy="steps",
            eval_steps=5000,
            logging_steps=100,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=4,
            learning_rate=1e-4,
            weight_decay=0.01,
            predict_with_generate=True,
            fp16=torch.cuda.is_available(),
            report_to="none",
            save_strategy="no"
        )

        # Initialize the Trainer
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=lambda eval_preds: compute_metrics(eval_preds, tokenizer),
        )

        # Train the model with LoRA
        trainer.train()
        
        # Evaluate on train set
        results = trainer.evaluate(eval_dataset=train_dataset)
        log(f"Results for LoRA Rank {r} on Train Set: {results}")

        # Evaluate on validation set
        results = trainer.evaluate()
        log(f"Results for LoRA Rank {r} on Validation Set: {results}")

        # Evaluate on test set
        results = trainer.evaluate(eval_dataset=test_dataset)
        log(f"Results for LoRA Rank {r} on Test Set: {results}")