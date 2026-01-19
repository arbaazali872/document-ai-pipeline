import csv
import json
import random
import subprocess
from pathlib import Path

class SROIEDataLoader:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.img_dir = self.data_dir / "img"
        self.box_dir = self.data_dir / "box"
        self.key_dir = self.data_dir / "key"
    
    @staticmethod
    def download_sroie_data(data_dir):
        """Download SROIE dataset via git"""
        data_path = Path(data_dir)
        
        if data_path.exists() and (data_path / "img").exists():
            print("✅ SROIE data already exists")
            return
        
        print("\n📥 Downloading SROIE dataset...")
        
        try:
            subprocess.run([
                "git", "clone", 
                "https://github.com/zzzDavid/ICDAR-2019-SROIE.git",
                "temp_sroie"
            ], check=True)
            
            import shutil
            shutil.move("temp_sroie/data", str(data_path))
            shutil.rmtree("temp_sroie")
            
            print("✅ Dataset ready\n")
            
        except Exception as e:
            print(f"❌ Auto-download failed: {e}")
            print("\n⚠️  MANUAL SETUP:")
            print("git clone https://github.com/zzzDavid/ICDAR-2019-SROIE.git")
            print("mv ICDAR-2019-SROIE/data ./sroie_data")
            raise
    
    def load_boxes(self, file_id):
        box_file = self.box_dir / f"{file_id}.csv"
        boxes, texts = [], []
        
        if box_file.exists():
            with open(box_file, 'r', encoding='utf-8') as f:
                for row in csv.reader(f):
                    if len(row) >= 9:
                        x_coords = [int(row[i]) for i in [0, 2, 4, 6]]
                        y_coords = [int(row[i]) for i in [1, 3, 5, 7]]
                        boxes.append([min(x_coords), min(y_coords), max(x_coords), max(y_coords)])
                        texts.append(row[8] if len(row) > 8 else "")
        
        return boxes, texts
    
    def load_entities(self, file_id):
        entity_file = self.key_dir / f"{file_id}.json"
        if entity_file.exists():
            with open(entity_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def get_dataset_split(self, split='train', seed=42):
        img_files = sorted([f.stem for f in self.img_dir.glob("*.jpg")])
        random.seed(seed)
        random.shuffle(img_files)
        
        split_idx = int(len(img_files) * 0.8)
        return img_files[:split_idx] if split == 'train' else img_files[split_idx:]
    
    def process_dataset(self, split='train', max_samples=None):
        file_ids = self.get_dataset_split(split)
        if max_samples:
            file_ids = file_ids[:max_samples]
        
        processed = []
        for file_id in file_ids:
            img_path = self.img_dir / f"{file_id}.jpg"
            if not img_path.exists():
                continue
            
            boxes, texts = self.load_boxes(file_id)
            entities = self.load_entities(file_id)
            
            processed.append({
                'file_id': file_id,
                'image_path': str(img_path),
                'boxes': boxes,
                'texts': texts,
                'entities': entities
            })
        
        print(f"📊 {split.upper()}: {len(processed)} samples")
        return processed