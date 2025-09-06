from adapters import AutoAdapterModel, AdapterConfig
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)

import torch
from utils import compute_metrics
from log import log

def adapter_tuning(tokenizer, train_dataset, test_dataset, eval_dataset):
    
    """
    Fine-tunes T5 using adapter tuning with varying reduction factors.

    Args:
        tokenizer (Tokenizer): Tokenizer used for preprocessing.
        train_dataset (Dataset): Training dataset.
        test_dataset (Dataset): Test dataset.
        eval_dataset (Dataset): Validation dataset.
    """
    
    # Setup data collator for seq2seq
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=None)
    adapter_reduction_factors = [4, 8, 16, 32, 64, 128, 256]

    for r in adapter_reduction_factors:
        log(f"\n\n\n=== Training with adapter reduction factor = {r}\n\n\n")

        # Load base T5 model with adapter support
        model = AutoAdapterModel.from_pretrained("t5-small")

        # Configure and add adapter
        adapter_name = f"imdb_adapter_r{r}"
        adapter_config = AdapterConfig.load("houlsby", reduction_factor=r)
        model.add_adapter(adapter_name, config=adapter_config)
        model.train_adapter(adapter_name)
        model.set_active_adapters(adapter_name)

        # Update collator with model
        data_collator.model = model

        # Log parameter counts
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        log(f"Total parameters: {total_params:,}")
        log(f"Trainable parameters (adapter only): {trainable_params:,}")
        log(f"Adapter fraction: {trainable_params / total_params * 100:.4f} %")

        # Set training arguments
        training_args = Seq2SeqTrainingArguments(
            output_dir=f"t5_adapter_imdb_r{r}",
            eval_strategy="steps",
            eval_steps=5000,
            logging_steps=200,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=4,
            learning_rate=1e-4,
            weight_decay=0.01,
            predict_with_generate=True,
            fp16=torch.cuda.is_available(),
            save_strategy="no",
            report_to="none"
        )

        # Initialize trainer
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=lambda eval_preds: compute_metrics(eval_preds, tokenizer),
        )

        # Training
        trainer.train()

        # Evaluation on train set
        results = trainer.evaluate(eval_dataset=train_dataset)
        log(f"Results for Adapter Size reduction {r} on Train Set: {results}")

        # Evaluation on validation set
        results = trainer.evaluate()
        log(f"Results for Adapter Size reduction {r} on Validation Set: {results}")

        # Evaluation on test set
        results = trainer.evaluate(eval_dataset=test_dataset)
        log(f"Results for Adapter Size reduction {r} on Test Set: {results}")