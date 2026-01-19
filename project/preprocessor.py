import torch
import pickle
from pathlib import Path
from PIL import Image
from transformers import LayoutLMv3Processor

class InputPreparation:
    def __init__(self, model_name, max_length):
        self.processor = LayoutLMv3Processor.from_pretrained(model_name, apply_ocr=False)
        self.max_length = max_length
        self.label_map = {
            'O': 0, 'B-COMPANY': 1, 'I-COMPANY': 2,
            'B-DATE': 3, 'I-DATE': 4, 'B-ADDRESS': 5,
            'I-ADDRESS': 6, 'B-TOTAL': 7, 'I-TOTAL': 8
        }
    
    def normalize_box(self, box, width, height):
        return [
            min(max(int(1000 * box[0] / width), 0), 1000),
            min(max(int(1000 * box[1] / height), 0), 1000),
            min(max(int(1000 * box[2] / width), 0), 1000),
            min(max(int(1000 * box[3] / height), 0), 1000)
        ]
    
    def normalize_text(self, text):
        return text.strip().lower().replace(" ", "")
    
    def create_labels(self, texts, entities, encoding):
        word_ids = encoding.word_ids(batch_index=0)
        labels = [self.label_map['O']] * len(word_ids)
        
        normalized_entities = {k.upper(): self.normalize_text(v) for k, v in entities.items()}
        normalized_texts = [self.normalize_text(t) for t in texts]
        
        entity_word_map = {}
        for entity_type, entity_value_norm in normalized_entities.items():
            if len(entity_value_norm) < 3:
                continue
            
            for word_idx, text_norm in enumerate(normalized_texts):
                if text_norm == entity_value_norm:
                    entity_word_map.setdefault(word_idx, []).append(entity_type)
                elif len(entity_value_norm) > 5 and len(text_norm) > 3:
                    if text_norm in entity_value_norm or entity_value_norm in text_norm:
                        entity_word_map.setdefault(word_idx, []).append(entity_type)
        
        for token_idx, word_idx in enumerate(word_ids):
            if word_idx is None:
                labels[token_idx] = -100
            elif word_idx in entity_word_map:
                entity_type = entity_word_map[word_idx][0]
                if token_idx == 0 or word_ids[token_idx - 1] != word_idx:
                    labels[token_idx] = self.label_map[f'B-{entity_type}']
                else:
                    labels[token_idx] = self.label_map[f'I-{entity_type}']
        
        return torch.tensor(labels)
    
    def prepare_sample(self, sample):
        image = Image.open(sample['image_path']).convert("RGB")
        width, height = image.size
        
        texts = [t for t in sample['texts'] if t and t.strip()]
        boxes = [b for t, b in zip(sample['texts'], sample['boxes']) if t and t.strip()]
        
        if not texts:
            return None
        
        normalized_boxes = [self.normalize_box(b, width, height) for b in boxes]
        
        encoding = self.processor(
            image, texts, boxes=normalized_boxes,
            truncation=True, padding="max_length",
            max_length=self.max_length, return_tensors="pt"
        )
        
        labels = self.create_labels(texts, sample['entities'], encoding)
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'bbox': encoding['bbox'].squeeze(0),
            'pixel_values': encoding['pixel_values'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': labels
        }
    
    def prepare_dataset(self, samples, cache_file, cache_dir):
        """Prepare dataset with disk caching"""
        cache_path = Path(cache_dir) / cache_file
        
        if cache_path.exists():
            print(f"📦 Loading from cache: {cache_file}")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        
        print(f"⚙️  Processing (will cache to {cache_file})...")
        prepared = []
        for i, sample in enumerate(samples):
            if i % 50 == 0:
                print(f"  Processed {i}/{len(samples)}...")
            try:
                result = self.prepare_sample(sample)
                if result:
                    prepared.append(result)
            except Exception as e:
                print(f"⚠️  Error: {sample['file_id']}: {e}")
        
        with open(cache_path, 'wb') as f:
            pickle.dump(prepared, f)
        print(f"💾 Cached to disk: {cache_file}")
        
        return prepared