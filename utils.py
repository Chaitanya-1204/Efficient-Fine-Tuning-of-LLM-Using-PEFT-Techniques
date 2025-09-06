from transformers import DataCollatorForSeq2Seq 
import torch




def compute_metrics(eval_preds , tokenizer):
    preds, labels = eval_preds
    
    # Check if preds are logits, and convert to token IDs
    if isinstance(preds, tuple) or len(preds.shape) > 2:
        preds = preds.argmax(-1)
    
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    pred_labels = [1 if pred.strip() == "positive" else 0 for pred in decoded_preds]
    true_labels = [1 if lab.strip() == "positive" else 0 for lab in decoded_labels]

    correct = sum(p == t for p, t in zip(pred_labels, true_labels))
    accuracy = correct / len(true_labels) if true_labels else 0.0
    return {"accuracy": accuracy}