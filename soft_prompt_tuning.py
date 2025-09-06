# loading Libraries
from transformers import (
    T5Tokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    T5ForConditionalGeneration
)
from peft import PromptTuningConfig, get_peft_model, TaskType
import torch


from utils import compute_metrics
from log import log 

def soft_prompt_tuning(tokenizer, device, train_dataset, eval_dataset, test_dataset):
    
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=None)
    virtual_token_counts = [5 , 10 ,  20 , 30 , 40 , 50]

    for vt in virtual_token_counts:
        log(f"\n\n=== Soft-Prompt Tuning with {vt} virtual tokens ===\n\n")

        # Load tokenizer and model
        model_name = "t5-small"
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)

        # Create and apply Prompt Tuning config
        prompt_config = PromptTuningConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            num_virtual_tokens=vt,
        )
        model = get_peft_model(model, prompt_config)


        # print parameter stats
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log(f"[vt={vt}] Total: {total_params:,}, Trainable: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")

        


        #  Update data collator
        data_collator.model = model

        # Training arguments
        training_args = Seq2SeqTrainingArguments(
            output_dir=f"t5_prompt_imdb_{vt}tok",
            eval_strategy="steps",
            
            eval_steps=5000 ,
            logging_steps=100,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=3,
            learning_rate=5e-3,
            weight_decay=0.01,
            predict_with_generate=True,
            fp16=torch.cuda.is_available(),
            save_strategy="no",            
            report_to="none"
        )

        # Initialize Trainer
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=lambda eval_preds: compute_metrics(eval_preds, tokenizer),
        
        )

        # Train and evaluate
        trainer.train()
        
        # Evaluation on train Set
        results = trainer.evaluate(eval_dataset = train_dataset)
        log(f"Results for {vt} virtual tokens: {results}")
        
        # Evaluate on Validation Set 
        
        results = trainer.evaluate()
        log(f"Results for {vt} virtual tokens: {results}")
        
        # Evaluation on Test Set

        results = trainer.evaluate(eval_dataset = test_dataset)
        log(f"Results for {vt} virtual tokens: {results}")
