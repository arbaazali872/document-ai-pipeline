import torch
import numpy as np
from transformers import (
    LayoutLMv3ForTokenClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from datasets import load_metric
from utils import CostMonitorCallback

class SROIETrainer:
    def __init__(self, config, timer):
        self.config = config
        self.timer = timer
        self.label_map = {
            'O': 0, 'B-COMPANY': 1, 'I-COMPANY': 2,
            'B-DATE': 3, 'I-DATE': 4, 'B-ADDRESS': 5,
            'I-ADDRESS': 6, 'B-TOTAL': 7, 'I-TOTAL': 8
        }
        self.id2label = {v: k for k, v in self.label_map.items()}
        self.label_list = list(self.label_map.keys())
        self.metric = load_metric("seqeval")
    
    def create_model(self):
        print("\n🤖 Loading model...")
        return LayoutLMv3ForTokenClassification.from_pretrained(
            self.config.MODEL_NAME,
            id2label=self.id2label,
            label2id=self.label_map
        )
    
    def compute_metrics(self, p):
        preds = np.argmax(p.predictions, axis=2)
        true_preds = [[self.label_list[p] for p, l in zip(pred, lab) if l != -100]
                     for pred, lab in zip(preds, p.label_ids)]
        true_labels = [[self.label_list[l] for p, l in zip(pred, lab) if l != -100]
                      for pred, lab in zip(preds, p.label_ids)]
        results = self.metric.compute(predictions=true_preds, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"]
        }
    
    def collate_fn(self, features):
        return {
            'input_ids': torch.stack([f['input_ids'] for f in features]),
            'bbox': torch.stack([f['bbox'] for f in features]),
            'pixel_values': torch.stack([f['pixel_values'] for f in features]),
            'attention_mask': torch.stack([f['attention_mask'] for f in features]),
            'labels': torch.stack([f['labels'] for f in features])
        }
    
    def train(self, train_dataset, eval_dataset):
        model = self.create_model()
        
        training_args = TrainingArguments(
            output_dir=self.config.OUTPUT_DIR,
            num_train_epochs=self.config.NUM_EPOCHS,
            per_device_train_batch_size=self.config.BATCH_SIZE,
            per_device_eval_batch_size=self.config.BATCH_SIZE,
            gradient_accumulation_steps=self.config.GRADIENT_ACCUMULATION,
            learning_rate=self.config.LEARNING_RATE,
            eval_strategy="steps",
            eval_steps=self.config.EVAL_STEPS,
            save_steps=self.config.SAVE_STEPS,
            logging_steps=self.config.LOGGING_STEPS,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            report_to="none",
            remove_unused_columns=False,
            fp16=True,
            push_to_hub=False,  # We'll push manually after training
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=self.collate_fn,
            compute_metrics=self.compute_metrics,
            callbacks=[
                CostMonitorCallback(self.timer),
                EarlyStoppingCallback(early_stopping_patience=self.config.EARLY_STOPPING_PATIENCE)
            ]
        )
        
        print("\n🎯 Training started...")
        print(f"⚡ Effective batch size: {self.config.BATCH_SIZE * self.config.GRADIENT_ACCUMULATION}")
        print(f"📈 Total epochs: {self.config.NUM_EPOCHS}")
        print(f"💾 Checkpoints: every {self.config.SAVE_STEPS} steps")
        
        trainer.train()
        
        print("\n📊 Final evaluation...")
        results = trainer.evaluate()
        
        print("\n" + "="*80)
        print("✅ TRAINING COMPLETE")
        print("="*80)
        for key, value in results.items():
            print(f"  {key}: {value:.4f}")
        
        trainer.save_model(f"{self.config.OUTPUT_DIR}/final")
        print(f"\n💾 Model saved to: {self.config.OUTPUT_DIR}/final")
        
        elapsed = self.timer.check()
        cost = elapsed * 0.34
        print(f"\n💰 Final cost: ${cost:.2f} ({elapsed:.2f} hours)")
        
        return trainer, results