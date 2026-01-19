import os
import json
import csv
from PIL import Image
from pathlib import Path

class SROIEDataLoader:
    def __init__(self, data_dir):
        """
        data_dir structure:
        - data_dir/
          - img/      (.jpg files)
          - box/      (.csv files with coordinates)
          - key/      (.json files with entities)
        """
        self.data_dir = Path(data_dir)
        self.img_dir = self.data_dir / "img"
        self.box_dir = self.data_dir / "box"
        self.key_dir = self.data_dir / "key"
        
    def load_boxes(self, file_id):
        """Parse bounding box CSV file"""
        box_file = self.box_dir / f"{file_id}.csv"
        boxes = []
        texts = []
        
        if not box_file.exists():
            return boxes, texts
        
        try:
            with open(box_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 9:  # x1,y1,x2,y2,x3,y3,x4,y4,text
                        # Convert to [x_min, y_min, x_max, y_max]
                        x_coords = [int(row[i]) for i in [0, 2, 4, 6]]
                        y_coords = [int(row[i]) for i in [1, 3, 5, 7]]
                        
                        box = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
                        text = row[8] if len(row) > 8 else ""
                        
                        boxes.append(box)
                        texts.append(text)
        except Exception as e:
            print(f"Error loading boxes for {file_id}: {e}")
        
        return boxes, texts
    
    def load_entities(self, file_id):
        """Load entity annotations from JSON"""
        entity_file = self.key_dir / f"{file_id}.json"
        
        if not entity_file.exists():
            return {}
        
        try:
            with open(entity_file, 'r', encoding='utf-8') as f:
                entities = json.load(f)
            return entities
        except Exception as e:
            print(f"Error loading entities for {file_id}: {e}")
            return {}
    
    def get_dataset_split(self, split='train', shuffle=True, seed=42):
        """Get file IDs with proper train/test split"""
        img_files = sorted([f.stem for f in self.img_dir.glob("*.jpg")])
        
        if shuffle:
            import random
            random.seed(seed)
            random.shuffle(img_files)
        
        # 80-20 split
        split_idx = int(len(img_files) * 0.8)
        
        if split == 'train':
            return img_files[:split_idx]
        else:
            return img_files[split_idx:]
    
    def process_dataset(self, split='train', max_samples=None):
        """Process dataset split - stores file paths only"""
        file_ids = self.get_dataset_split(split)
        
        if max_samples:
            file_ids = file_ids[:max_samples]
        
        processed_data = []
        
        for file_id in file_ids:
            img_path = self.img_dir / f"{file_id}.jpg"
            
            if not img_path.exists():
                print(f"Image not found: {img_path}")
                continue
            
            boxes, texts = self.load_boxes(file_id)
            entities = self.load_entities(file_id)
            
            processed_data.append({
                'file_id': file_id,
                'image_path': str(img_path),  # Store path, not image
                'boxes': boxes,
                'texts': texts,
                'entities': entities
            })
        
        print(f"\n{split.upper()} Split Summary:")
        print(f"Total files: {len(file_ids)}")
        print(f"Processed: {len(processed_data)}")
        
        return processed_data


# Test Stage 1
if __name__ == "__main__":
    data_dir = "./sroie_data"
    
    loader = SROIEDataLoader(data_dir)
    
    # Test on small sample
    train_data = loader.process_dataset(split='train', max_samples=10)
    
    # Display sample
    if train_data:
        sample = train_data[0]
        print(f"\nSample Data:")
        print(f"File ID: {sample['file_id']}")
        print(f"Image path: {sample['image_path']}")
        print(f"Number of boxes: {len(sample['boxes'])}")
        print(f"Number of texts: {len(sample['texts'])}")
        print(f"Entities: {sample['entities']}")
        
        if sample['boxes']:
            print(f"First box: {sample['boxes'][0]}")
            print(f"First text: {sample['texts'][0]}")