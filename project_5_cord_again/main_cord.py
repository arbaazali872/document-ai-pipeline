"""
NeuroKnow AI - Problem 1: Document Understanding with CORD Dataset

Simple pipeline using CORD (Consolidated Receipt Dataset) with pre-trained model.
"""

import os
import json
import time
from pathlib import Path
from collections import defaultdict

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForTokenClassification


# ============================================================================
# CONFIGURATION
# ============================================================================

# Path to CORD dataset (folder containing dev/, test/, train/)
CORD_DATASET_PATH = "D:\my_projects\document-ai-pipeline\datasets\CORD"  # UPDATE THIS: r"D:\path\to\CORD"

# Pre-trained model fine-tuned on CORD
MODEL_NAME = "nielsr/layoutlmv3-finetuned-cord"

OUTPUT_DIR = Path("results_cord")
EXAMPLES_DIR = Path("examples_cord")
NUM_TEST_SAMPLES = 100


# CORD label mapping
LABEL_MAP = {
    0: "O",
    1: "B-MENU.NM",
    2: "B-MENU.NUM",
    3: "B-MENU.UNITPRICE",
    4: "B-MENU.CNT",
    5: "B-MENU.DISCOUNTPRICE",
    6: "B-MENU.PRICE",
    7: "B-MENU.ITEMSUBTOTAL",
    8: "B-MENU.VATYN",
    9: "B-MENU.ETC",
    10: "B-VOID_MENU.NM",
    11: "B-VOID_MENU.PRICE",
    12: "B-SUB_TOTAL.SUBTOTAL_PRICE",
    13: "B-SUB_TOTAL.DISCOUNT_PRICE",
    14: "B-SUB_TOTAL.SERVICE_PRICE",
    15: "B-SUB_TOTAL.OTHERSVC_PRICE",
    16: "B-SUB_TOTAL.TAX_PRICE",
    17: "B-SUB_TOTAL.ETC",
    18: "B-TOTAL.TOTAL_PRICE",
    19: "B-TOTAL.TOTAL_ETC",
    20: "B-TOTAL.CASHPRICE",
    21: "B-TOTAL.CHANGEPRICE",
    22: "B-TOTAL.CREDITCARDPRICE",
    23: "B-TOTAL.EMONEYPRICE",
    24: "B-TOTAL.MENUTYPE_CNT",
    25: "B-TOTAL.MENUQTY_CNT",
    26: "I-MENU.NM",
    27: "I-MENU.NUM",
    28: "I-MENU.UNITPRICE",
    29: "I-MENU.CNT",
    30: "I-MENU.DISCOUNTPRICE",
    31: "I-MENU.PRICE",
    32: "I-MENU.ITEMSUBTOTAL",
    33: "I-MENU.VATYN",
    34: "I-MENU.ETC",
    35: "I-VOID_MENU.NM",
    36: "I-VOID_MENU.PRICE",
    37: "I-SUB_TOTAL.SUBTOTAL_PRICE",
    38: "I-SUB_TOTAL.DISCOUNT_PRICE",
    39: "I-SUB_TOTAL.SERVICE_PRICE",
    40: "I-SUB_TOTAL.OTHERSVC_PRICE",
    41: "I-SUB_TOTAL.TAX_PRICE",
    42: "I-SUB_TOTAL.ETC",
    43: "I-TOTAL.TOTAL_PRICE",
    44: "I-TOTAL.TOTAL_ETC",
    45: "I-TOTAL.CASHPRICE",
    46: "I-TOTAL.CHANGEPRICE",
    47: "I-TOTAL.CREDITCARDPRICE",
    48: "I-TOTAL.EMONEYPRICE",
    49: "I-TOTAL.MENUTYPE_CNT",
    50: "I-TOTAL.MENUQTY_CNT"
}

# Reverse mapping
ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}


# ============================================================================
# STEP 1: LOAD MODEL
# ============================================================================

def load_model():
    """Load pre-trained LayoutLMv3 fine-tuned on CORD."""
    print("Loading pre-trained model...")
    print(f"Model: {MODEL_NAME}")
    
    processor = AutoProcessor.from_pretrained(MODEL_NAME, apply_ocr=False)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    print(f"Model loaded on device: {device}")
    return model, processor, device


# ============================================================================
# STEP 2: LOAD CORD DATASET
# ============================================================================

def load_cord_dataset(data_dir):
    """
    Load CORD dataset.
    
    Structure:
    data_dir/
    ├── dev/
    │   ├── image/
    │   └── json/
    ├── test/
    │   ├── image/
    │   └── json/
    └── train/
        ├── image/
        └── json/
    
    Args:
        data_dir: Path to CORD dataset root
    
    Returns:
        dict with 'train', 'dev', 'test' splits
    """
    data_dir = Path(data_dir)
    
    def load_split(split_name):
        """Load a single split."""
        split_dir = data_dir / split_name
        img_dir = split_dir / "image"
        json_dir = split_dir / "json"
        
        dataset = []
        
        json_files = sorted(json_dir.glob("*.json"))
        
        for json_file in json_files:
            file_id = json_file.stem
            img_path = img_dir / f"{file_id}.png"
            
            if not img_path.exists():
                continue
            
            # Load annotation
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract words, boxes, labels
            words = []
            boxes = []
            labels = []
            
            for line in data.get("valid_line", []):
                category = line.get("category", "O")
                
                # Convert category to BIO format
                # e.g., "total.total_price" → "TOTAL.TOTAL_PRICE"
                if category != "O":
                    category_upper = category.upper()
                else:
                    category_upper = "O"
                
                line_words = line.get("words", [])
                
                for idx, word_info in enumerate(line_words):
                    words.append(word_info["text"])
                    
                    # Extract bounding box from quad
                    quad = word_info["quad"]
                    x_coords = [quad["x1"], quad["x2"], quad["x3"], quad["x4"]]
                    y_coords = [quad["y1"], quad["y2"], quad["y3"], quad["y4"]]
                    xmin = min(x_coords)
                    ymin = min(y_coords)
                    xmax = max(x_coords)
                    ymax = max(y_coords)
                    boxes.append([xmin, ymin, xmax, ymax])
                    
                    # Assign BIO label
                    if category_upper == "O":
                        labels.append("O")
                    elif idx == 0:
                        # First word gets B- (Beginning)
                        labels.append(f"B-{category_upper}")
                    else:
                        # Subsequent words get I- (Inside)
                        labels.append(f"I-{category_upper}")
            
            sample = {
                'id': file_id,
                'image_path': str(img_path),
                'words': words,
                'bboxes': boxes,
                'labels': labels
            }
            
            dataset.append(sample)
        
        return dataset
    
    # Load all splits
    train_data = load_split("train") if (data_dir / "train").exists() else []
    dev_data = load_split("dev") if (data_dir / "dev").exists() else []
    test_data = load_split("test") if (data_dir / "test").exists() else []
    
    print(f"Loaded CORD dataset:")
    print(f"  Train: {len(train_data)} samples")
    print(f"  Dev: {len(dev_data)} samples")
    print(f"  Test: {len(test_data)} samples")
    
    return {
        'train': train_data,
        'dev': dev_data,
        'test': test_data
    }


# ============================================================================
# STEP 3: PREPROCESSING
# ============================================================================

def normalize_box(box, width, height):
    """Normalize box to 0-1000 range and clip to valid range."""
    return [
        max(0, min(1000, int(1000 * box[0] / width))),
        max(0, min(1000, int(1000 * box[1] / height))),
        max(0, min(1000, int(1000 * box[2] / width))),
        max(0, min(1000, int(1000 * box[3] / height))),
    ]


def preprocess_document(image, words, boxes, processor, device):
    """Preprocess document for model input."""
    width, height = image.size
    
    # Normalize boxes
    normalized_boxes = [normalize_box(box, width, height) for box in boxes]
    
    # Prepare inputs
    encoding = processor(
        image,
        words,
        boxes=normalized_boxes,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=512
    )
    
    # Move to device
    encoding = {k: v.to(device) for k, v in encoding.items()}
    
    return encoding


# ============================================================================
# STEP 4: ENTITY EXTRACTION
# ============================================================================

def extract_entities(outputs, words):
    """Extract entities from model predictions using BIO tagging."""
    predictions = torch.argmax(outputs.logits, dim=-1).squeeze().cpu().numpy()
    
    entities = defaultdict(list)
    current_entity = None
    current_text = []
    
    for idx, (word, pred_id) in enumerate(zip(words, predictions)):
        if idx >= len(predictions):
            break
        
        pred_label = LABEL_MAP.get(pred_id, "O")
        
        if pred_label.startswith("B-"):
            # Save previous entity if exists
            if current_entity and current_text:
                entities[current_entity].append(" ".join(current_text))
            
            # Start new entity
            current_entity = pred_label[2:]  # Remove "B-"
            current_text = [word]
            
        elif pred_label.startswith("I-"):
            # Continue current entity
            entity_type = pred_label[2:]  # Remove "I-"
            if current_entity == entity_type:
                current_text.append(word)
            else:
                # Mismatched I- tag, treat as new entity
                if current_entity and current_text:
                    entities[current_entity].append(" ".join(current_text))
                current_entity = entity_type
                current_text = [word]
        else:
            # O tag - save current entity if exists
            if current_entity and current_text:
                entities[current_entity].append(" ".join(current_text))
                current_entity = None
                current_text = []
    
    # Save last entity
    if current_entity and current_text:
        entities[current_entity].append(" ".join(current_text))
    
    # Convert lists to strings (join multiple instances with " | ")
    result = {}
    for k, v in entities.items():
        if len(v) == 1:
            result[k] = v[0]
        else:
            result[k] = " | ".join(v)  # Multiple items separated by |
    
    return result


# ============================================================================
# STEP 5: PROCESS DOCUMENT
# ============================================================================

def process_document(sample, model, processor, device):
    """Process single document."""
    start_time = time.time()
    
    # Load image
    image_path = sample["image_path"]
    image = Image.open(image_path).convert("RGB")
    words = sample["words"]
    boxes = sample["bboxes"]
    
    # Preprocess
    inputs = preprocess_document(image, words, boxes, processor, device)
    
    # Run model
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Extract entities
    entities = extract_entities(outputs, words)
    
    # Create result
    result = {
        "document_id": sample.get("id", "unknown"),
        "extracted_fields": entities,
        "metadata": {
            "extraction_method": "LayoutLMv3-CORD"
        }
    }
    
    processing_time = time.time() - start_time
    
    return result, processing_time


# ============================================================================
# STEP 6: EVALUATION
# ============================================================================

def evaluate_results(predictions, ground_truths):
    """Simple evaluation - check if key fields extracted."""
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    # Count how many documents have key fields extracted
    total_docs = len(predictions)
    
    key_fields = ["TOTAL.TOTAL_PRICE", "MENU.NM", "MENU.PRICE"]
    field_counts = defaultdict(int)
    
    for pred in predictions:
        extracted = pred.get("extracted_fields", {})
        for field in key_fields:
            if field in extracted and extracted[field]:
                field_counts[field] += 1
    
    print("\nKey Fields Extracted:")
    for field in key_fields:
        count = field_counts[field]
        pct = (count / total_docs * 100) if total_docs > 0 else 0
        print(f"  {field}: {count}/{total_docs} ({pct:.1f}%)")
    
    return {
        "total_documents": total_docs,
        "field_extraction_rates": {k: v/total_docs for k, v in field_counts.items()}
    }


# ============================================================================
# STEP 7: SAVE RESULTS
# ============================================================================

def save_results(predictions, metrics, processing_times):
    """Save results."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    with open(OUTPUT_DIR / "predictions.json", "w") as f:
        json.dump(predictions, f, indent=2)
    
    avg_time = sum(processing_times) / len(processing_times) if processing_times else 0
    
    results = {
        "metrics": metrics,
        "performance": {
            "avg_processing_time_sec": avg_time,
            "total_documents": len(predictions)
        }
    }
    
    with open(OUTPUT_DIR / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {OUTPUT_DIR}/")


def save_examples(predictions, num_examples=5):
    """Save examples."""
    EXAMPLES_DIR.mkdir(exist_ok=True)
    
    examples = predictions[:num_examples]
    
    with open(EXAMPLES_DIR / "examples.json", "w") as f:
        json.dump(examples, f, indent=2)
    
    print(f"Examples saved to {EXAMPLES_DIR}/examples.json")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Main execution."""
    print("="*60)
    print("NeuroKnow AI - Problem 1: CORD Pipeline")
    print("="*60)
    
    if not CORD_DATASET_PATH:
        print("\n❌ ERROR: Set CORD_DATASET_PATH in script")
        print("Example: CORD_DATASET_PATH = r'D:\\path\\to\\CORD'")
        return
    
    # Load model
    model, processor, device = load_model()
    
    # Load dataset
    dataset = load_cord_dataset(CORD_DATASET_PATH)
    test_data = dataset["train"]
    
    if not test_data:
        print("❌ No test data found!")
        return
    
    # Process documents
    num_samples = min(NUM_TEST_SAMPLES, len(test_data))
    print(f"\nProcessing {num_samples} test samples...\n")
    
    predictions = []
    processing_times = []
    
    for i in range(num_samples):
        if i % 10 == 0:
            print(f"Processing {i+1}/{num_samples}...")
        
        sample = test_data[i]
        result, proc_time = process_document(sample, model, processor, device)
        
        predictions.append(result)
        processing_times.append(proc_time)
    
    # Evaluate
    metrics = evaluate_results(predictions, [])
    
    # Save
    save_results(predictions, metrics, processing_times)
    save_examples(predictions, num_examples=5)
    
    print("\n" + "="*60)
    print("Pipeline completed!")
    print("="*60)


if __name__ == "__main__":
    main()