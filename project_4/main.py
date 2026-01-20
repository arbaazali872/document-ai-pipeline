"""
NeuroKnow AI - Problem 1: Multimodal Document Understanding
SROIE (Scanned Receipts OCR and Information Extraction) Pipeline

Simple functional pipeline for extracting key-value pairs from receipts.
"""

import os
import json
import time
from typing import Dict, List, Tuple
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForTokenClassification
from datasets import load_dataset
import numpy as np
from collections import defaultdict


# ============================================================================
# CONFIGURATION
# ============================================================================

# Set this to your local ICDAR-2019-SROIE/data path if you have it downloaded
# Use raw string r"path" or forward slashes "path/to/data"
# Example: r"D:\my_projects\ICDAR-2019-SROIE\data" or "D:/my_projects/ICDAR-2019-SROIE/data"
LOCAL_SROIE_PATH = "D:\my_projects\document-ai-pipeline\sroie_data"  # UPDATE THIS PATH

MODEL_NAME = "Theivaprakasham/layoutlmv3-finetuned-sroie"
OUTPUT_DIR = Path("results")
EXAMPLES_DIR = Path("examples")
NUM_TEST_SAMPLES = 100  # Evaluate on first 100 test samples

# SROIE entity labels
ENTITY_LABELS = {
    0: "O",        # Outside
    1: "B-COMPANY",
    2: "I-COMPANY", 
    3: "B-DATE",
    4: "I-DATE",
    5: "B-ADDRESS",
    6: "I-ADDRESS",
    7: "B-TOTAL",
    8: "I-TOTAL"
}

# Reverse mapping
LABEL_TO_ID = {v: k for k, v in ENTITY_LABELS.items()}


# ============================================================================
# STEP 1: LOAD MODEL AND PROCESSOR
# ============================================================================

def load_model_and_processor():
    """
    Load pre-trained LayoutLMv3 model fine-tuned on SROIE dataset.
    
    Returns:
        tuple: (model, processor)
    """
    print("Loading pre-trained model...")
    print(f"Model: {MODEL_NAME}")
    
    processor = AutoProcessor.from_pretrained(MODEL_NAME, apply_ocr=False)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    print(f"Model loaded on device: {device}")
    return model, processor, device


# ============================================================================
# STEP 2: LOAD SROIE DATASET
# ============================================================================

def load_local_sroie(data_dir):
    """
    Load SROIE dataset from local ICDAR-2019-SROIE folder.
    
    Args:
        data_dir: Path to ICDAR-2019-SROIE/data folder
    
    Returns:
        dict with 'train' and 'test' datasets
    """
    import csv
    data_dir = Path(data_dir)
    box_dir = data_dir / "box"
    img_dir = data_dir / "img"
    key_dir = data_dir / "key"
    
    dataset = []
    
    # Get all CSV files in box directory
    box_files = sorted(box_dir.glob("*.csv"))
    
    for box_file in box_files:
        file_id = box_file.stem
        
        # Paths
        img_path = img_dir / f"{file_id}.jpg"
        key_path = key_dir / f"{file_id}.json"
        
        if not img_path.exists() or not key_path.exists():
            continue
        
        # Read bounding boxes and text from CSV
        words = []
        bboxes = []
        
        with open(box_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 9:
                    continue
                
                # Format: x1,y1,x2,y2,x3,y3,x4,y4,text
                x1, y1, x2, y2, x3, y3, x4, y4 = map(int, row[:8])
                text = ','.join(row[8:]).strip()
                
                if text:
                    words.append(text)
                    # Use top-left and bottom-right for bbox
                    bboxes.append([x1, y1, x3, y3])
        
        # Read ground truth entities from JSON
        with open(key_path, 'r', encoding='utf-8') as f:
            entities = json.load(f)
        
        # Create sample
        sample = {
            'id': file_id,
            'image_path': str(img_path),
            'words': words,
            'bboxes': bboxes,
            'entities': entities
        }
        
        dataset.append(sample)
    
    # Split into train/test
    # Use first 526 for train, last 100 for test
    train_data = dataset[:526]
    test_data = dataset[526:]
    
    print(f"Loaded local SROIE dataset:")
    print(f"  Train: {len(train_data)} samples")
    print(f"  Test: {len(test_data)} samples")
    
    return {
        'train': train_data,
        'test': test_data
    }


def load_sroie_dataset(local_path=None):
    """
    Load SROIE dataset - either from HuggingFace or local folder.
    
    Args:
        local_path: Path to ICDAR-2019-SROIE/data folder (if loading locally)
    
    Returns:
        dataset: Dataset object or dict
    """
    print("\nLoading SROIE dataset...")
    
    if local_path:
        # Load from local ICDAR-2019-SROIE folder
        dataset = load_local_sroie(local_path)
        return dataset
    else:
        # Load from HuggingFace
        try:
            from datasets import Features, Sequence, Value, Image as ImageFeature, ClassLabel
            
            dataset = load_dataset(
                "darentang/sroie",
                trust_remote_code=True
            )
            
            print(f"Dataset loaded successfully!")
            print(f"Train samples: {len(dataset['train'])}")
            print(f"Test samples: {len(dataset['test'])}")
            return dataset
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("\nAlternative: Load from local ICDAR-2019-SROIE folder")
            print("Set LOCAL_SROIE_PATH in the script")
            raise


# ============================================================================
# STEP 3: PREPROCESSING
# ============================================================================

def normalize_box(box, width, height):
    """
    Normalize bounding box coordinates to 0-1000 range (LayoutLM format).
    
    Args:
        box: [x0, y0, x1, y1] coordinates
        width: image width
        height: image height
    
    Returns:
        list: normalized box [x0, y0, x1, y1]
    """
    return [
        int(1000 * box[0] / width),
        int(1000 * box[1] / height),
        int(1000 * box[2] / width),
        int(1000 * box[3] / height),
    ]


def preprocess_document(image, words, boxes, processor, device):
    """
    Preprocess a single document for model input.
    
    Args:
        image: PIL Image
        words: list of OCR words
        boxes: list of bounding boxes (x0, y0, x1, y1)
        processor: HuggingFace processor
        device: torch device
    
    Returns:
        dict: model inputs
    """
    width, height = image.size
    
    # Normalize boxes to 0-1000 range
    normalized_boxes = [normalize_box(box, width, height) for box in boxes]
    
    # Prepare inputs for LayoutLM
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

def extract_ground_truth_from_tags(words, ner_tags):
    """
    Extract ground truth entities from NER tags.
    
    Args:
        words: list of words
        ner_tags: list of NER tag IDs
    
    Returns:
        dict: ground truth entities {company, date, address, total}
    """
    entities = {
        "company": "",
        "date": "",
        "address": "",
        "total": ""
    }
    
    current_entity = None
    current_text = []
    
    for word, tag_id in zip(words, ner_tags):
        tag_label = ENTITY_LABELS.get(tag_id, "O")
        
        if tag_label.startswith("B-"):
            # Save previous entity
            if current_entity and current_text:
                entity_key = current_entity.lower()
                entities[entity_key] = " ".join(current_text)
            
            # Start new entity
            current_entity = tag_label[2:]  # Remove "B-"
            current_text = [word]
            
        elif tag_label.startswith("I-"):
            # Continue current entity
            if current_entity == tag_label[2:]:
                current_text.append(word)
        else:
            # Outside tag - save current entity if exists
            if current_entity and current_text:
                entity_key = current_entity.lower()
                entities[entity_key] = " ".join(current_text)
                current_entity = None
                current_text = []
    
    # Save last entity
    if current_entity and current_text:
        entity_key = current_entity.lower()
        entities[entity_key] = " ".join(current_text)
    
    return entities


def extract_entities(model_output, words, boxes):
    """
    Extract entities from model predictions using BIO tagging.
    
    Args:
        model_output: model logits
        words: list of words
        boxes: list of bounding boxes
    
    Returns:
        dict: extracted entities {company, date, address, total}
    """
    # Get predictions
    predictions = torch.argmax(model_output.logits, dim=-1)
    predictions = predictions.squeeze().cpu().numpy()
    
    # Initialize entity dictionary
    entities = {
        "company": "",
        "date": "",
        "address": "",
        "total": ""
    }
    
    current_entity = None
    current_text = []
    
    # Process each token prediction
    for idx, (word, pred_id) in enumerate(zip(words, predictions)):
        if idx >= len(predictions):
            break
            
        pred_label = ENTITY_LABELS.get(pred_id, "O")
        
        if pred_label.startswith("B-"):
            # Save previous entity
            if current_entity and current_text:
                entity_key = current_entity.lower()
                entities[entity_key] = " ".join(current_text)
            
            # Start new entity
            current_entity = pred_label[2:]  # Remove "B-"
            current_text = [word]
            
        elif pred_label.startswith("I-"):
            # Continue current entity
            if current_entity == pred_label[2:]:
                current_text.append(word)
        else:
            # Outside tag - save current entity if exists
            if current_entity and current_text:
                entity_key = current_entity.lower()
                entities[entity_key] = " ".join(current_text)
                current_entity = None
                current_text = []
    
    # Save last entity
    if current_entity and current_text:
        entity_key = current_entity.lower()
        entities[entity_key] = " ".join(current_text)
    
    return entities


# ============================================================================
# STEP 5: PARSE TO STRUCTURED JSON
# ============================================================================

def parse_to_json(entities, doc_id="unknown"):
    """
    Convert extracted entities to structured JSON format.
    
    Args:
        entities: dict of extracted entities
        doc_id: document identifier
    
    Returns:
        dict: structured output
    """
    return {
        "document_id": doc_id,
        "extracted_fields": {
            "company": entities.get("company", ""),
            "date": entities.get("date", ""),
            "address": entities.get("address", ""),
            "total": entities.get("total", "")
        },
        "metadata": {
            "extraction_method": "LayoutLMv3"
        }
    }


# ============================================================================
# STEP 6: PROCESS SINGLE DOCUMENT
# ============================================================================

def process_single_document(sample, model, processor, device):
    """
    Complete pipeline for processing a single document.
    
    Args:
        sample: dataset sample
        model: LayoutLM model
        processor: processor
        device: torch device
    
    Returns:
        tuple: (extracted_json, processing_time)
    """
    start_time = time.time()
    
    # Extract data from sample
    # Handle local dataset (image_path string) and HuggingFace dataset
    image_path = ""
    doc_id = sample.get("id", "unknown")
    
    if "image" in sample:
        # HuggingFace dataset with actual images
        image = sample["image"]
        if not isinstance(image, Image.Image):
            image = Image.open(sample["image"]).convert("RGB")
    elif "image_path" in sample:
        # Local dataset or HuggingFace with paths
        image_path = sample["image_path"]
        image = Image.open(image_path).convert("RGB")
    else:
        raise ValueError("Sample must contain either 'image' or 'image_path'")
    
    words = sample["words"]
    boxes = sample["bboxes"]
    
    # Preprocess
    inputs = preprocess_document(image, words, boxes, processor, device)
    
    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Extract entities
    entities = extract_entities(outputs, words, boxes)
    
    # Parse to JSON with document ID
    result = parse_to_json(entities, doc_id)
    
    processing_time = time.time() - start_time
    
    return result, processing_time


# ============================================================================
# STEP 7: EVALUATION
# ============================================================================

def normalize_text(text):
    """Normalize text for comparison."""
    return text.lower().strip().replace(" ", "")


def calculate_field_accuracy(predicted, ground_truth, field_name):
    """
    Calculate exact match accuracy for a specific field.
    
    Args:
        predicted: predicted value
        ground_truth: ground truth value
        field_name: name of the field
    
    Returns:
        int: 1 if match, 0 otherwise
    """
    pred_norm = normalize_text(predicted)
    gt_norm = normalize_text(ground_truth)
    
    return 1 if pred_norm == gt_norm else 0


def evaluate_results(predictions, ground_truths):
    """
    Evaluate extraction performance.
    
    Args:
        predictions: list of predicted entities
        ground_truths: list of ground truth entities
    
    Returns:
        dict: evaluation metrics
    """
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    fields = ["company", "date", "address", "total"]
    field_scores = defaultdict(list)
    
    total_correct = 0
    total_fields = 0
    
    for pred, gt in zip(predictions, ground_truths):
        pred_fields = pred.get("extracted_fields", {})
        
        for field in fields:
            pred_value = pred_fields.get(field, "")
            gt_value = gt.get(field, "")
            
            score = calculate_field_accuracy(pred_value, gt_value, field)
            field_scores[field].append(score)
            
            total_correct += score
            total_fields += 1
    
    # Calculate metrics per field
    metrics = {}
    for field in fields:
        scores = field_scores[field]
        accuracy = sum(scores) / len(scores) if scores else 0
        metrics[field] = {
            "accuracy": accuracy,
            "correct": sum(scores),
            "total": len(scores)
        }
        print(f"\n{field.upper()}:")
        print(f"  Accuracy: {accuracy:.2%}")
        print(f"  Correct: {sum(scores)}/{len(scores)}")
    
    # Overall accuracy
    overall_accuracy = total_correct / total_fields if total_fields > 0 else 0
    metrics["overall"] = {
        "accuracy": overall_accuracy,
        "correct": total_correct,
        "total": total_fields
    }
    
    print(f"\nOVERALL ACCURACY: {overall_accuracy:.2%}")
    print(f"Total Correct Fields: {total_correct}/{total_fields}")
    
    return metrics


# ============================================================================
# STEP 8: SAVE RESULTS
# ============================================================================

def save_results(predictions, metrics, processing_times):
    """
    Save results to JSON files.
    
    Args:
        predictions: list of predictions
        metrics: evaluation metrics
        processing_times: list of processing times
    """
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Save predictions
    with open(OUTPUT_DIR / "predictions.json", "w") as f:
        json.dump(predictions, f, indent=2)
    
    # Save metrics
    avg_time = sum(processing_times) / len(processing_times) if processing_times else 0
    
    results = {
        "metrics": metrics,
        "performance": {
            "avg_processing_time_ms": avg_time * 1000,
            "total_documents": len(predictions),
            "documents_per_second": 1 / avg_time if avg_time > 0 else 0
        }
    }
    
    with open(OUTPUT_DIR / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {OUTPUT_DIR}/")
    print(f"  - predictions.json")
    print(f"  - metrics.json")


# ============================================================================
# STEP 9: VISUALIZE EXAMPLES
# ============================================================================

def save_example_visualizations(predictions, ground_truths, num_examples=5):
    """
    Save example predictions for visualization.
    
    Args:
        predictions: list of predictions
        ground_truths: list of ground truths
        num_examples: number of examples to save
    """
    EXAMPLES_DIR.mkdir(exist_ok=True)
    
    examples = []
    for i in range(min(num_examples, len(predictions))):
        example = {
            "document_id": predictions[i].get("document_id", f"doc_{i}"),
            "predicted": predictions[i]["extracted_fields"],
            "ground_truth": ground_truths[i],
            "match": {
                field: predictions[i]["extracted_fields"].get(field, "") == ground_truths[i].get(field, "")
                for field in ["company", "date", "address", "total"]
            }
        }
        examples.append(example)
    
    with open(EXAMPLES_DIR / "examples.json", "w") as f:
        json.dump(examples, f, indent=2)
    
    print(f"\nExample visualizations saved to {EXAMPLES_DIR}/examples.json")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """
    Main execution pipeline.
    """
    print("="*60)
    print("NeuroKnow AI - Problem 1: Document Understanding Pipeline")
    print("Dataset: SROIE (Scanned Receipts)")
    print("="*60)
    
    # Step 1: Load model
    model, processor, device = load_model_and_processor()
    
    # Step 2: Load dataset
    dataset = load_sroie_dataset(local_path=LOCAL_SROIE_PATH)
    test_data = dataset["test"]
    
    # Limit to NUM_TEST_SAMPLES
    num_samples = min(NUM_TEST_SAMPLES, len(test_data))
    print(f"\nProcessing {num_samples} test samples...")
    
    # Step 3-6: Process documents
    predictions = []
    ground_truths = []
    processing_times = []
    
    for i in range(num_samples):
        if i % 10 == 0:
            print(f"Processing document {i+1}/{num_samples}...")
        
        sample = test_data[i]
        
        # Extract ground truth - handle both formats
        if "entities" in sample:
            # Local dataset - entities already in dict format
            ground_truth = sample["entities"]
        elif "ner_tags" in sample:
            # HuggingFace dataset - extract from NER tags
            words = sample["words"]
            ner_tags = sample["ner_tags"]
            ground_truth = extract_ground_truth_from_tags(words, ner_tags)
        else:
            # No ground truth available
            ground_truth = {
                "company": "",
                "date": "",
                "address": "",
                "total": ""
            }
        
        # Process document
        result, proc_time = process_single_document(sample, model, processor, device)
        
        predictions.append(result)
        ground_truths.append(ground_truth)
        processing_times.append(proc_time)
    
    # Step 7: Evaluate
    metrics = evaluate_results(predictions, ground_truths)
    
    # Step 8: Save results
    save_results(predictions, metrics, processing_times)
    
    # Step 9: Visualize examples
    save_example_visualizations(predictions, ground_truths, num_examples=5)
    
    print("\n" + "="*60)
    print("Pipeline completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()