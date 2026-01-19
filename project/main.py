#!/usr/bin/env python3
import os
import sys
from huggingface_hub import HfApi, login
from config import Config
from data_loader import SROIEDataLoader
from preprocessor import InputPreparation
from trainer import SROIETrainer
from utils import TrainingTimer

def main():
    print("="*80)
    print("SROIE LAYOUTLMV3 TRAINING - RUNPOD RTX 2000")
    print("="*80)
    
    # Setup
    Config.setup_dirs()
    
    # Login to Hugging Face
    print("\n🔐 Logging into Hugging Face...")
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("❌ HF_TOKEN not found in environment variables!")
        print("Set it with: export HF_TOKEN=your_token_here")
        sys.exit(1)
    
    login(token=hf_token)
    print("✅ Logged in to Hugging Face")
    
    # Safety features summary
    print(f"\n⚙️  Safety features:")
    print(f"   - Cache preprocessing: ✅")
    print(f"   - Max length: {Config.MAX_LENGTH}")
    print(f"   - Eval steps: {Config.EVAL_STEPS}")
    print(f"   - Save steps: {Config.SAVE_STEPS}")
    print(f"   - Early stopping: {Config.EARLY_STOPPING_PATIENCE} evals")
    print(f"   - Auto-kill timer: {Config.MAX_TRAINING_HOURS}h")
    
    # Initialize timer
    timer = TrainingTimer(Config.MAX_TRAINING_HOURS)
    
    # Download data
    SROIEDataLoader.download_sroie_data(Config.DATA_DIR)
    
    # Load data
    print("\n📁 Loading data...")
    loader = SROIEDataLoader(Config.DATA_DIR)
    train_data = loader.process_dataset('train', Config.MAX_TRAIN_SAMPLES)
    eval_data = loader.process_dataset('test', Config.MAX_EVAL_SAMPLES)
    
    # Prepare inputs
    print("\n⚙️  Preparing inputs...")
    prep = InputPreparation(Config.MODEL_NAME, Config.MAX_LENGTH)
    train_dataset = prep.prepare_dataset(train_data, "train_cache.pkl", Config.CACHE_DIR)
    eval_dataset = prep.prepare_dataset(eval_data, "eval_cache.pkl", Config.CACHE_DIR)
    
    print(f"✅ Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")
    
    if len(train_dataset) == 0:
        print("❌ No training data prepared!")
        sys.exit(1)
    
    # Train
    trainer_obj = SROIETrainer(Config, timer)
    trainer, results = trainer_obj.train(train_dataset, eval_dataset)
    
    # Push to Hugging Face
    print(f"\n🚀 Pushing model to Hugging Face: {Config.HF_REPO_NAME}")
    try:
        trainer.push_to_hub(
            repo_id=Config.HF_REPO_NAME,
            commit_message=f"Training complete - F1: {results['eval_f1']:.4f}"
        )
        print(f"✅ Model pushed to: https://huggingface.co/{Config.HF_REPO_NAME}")
    except Exception as e:
        print(f"⚠️  Failed to push to hub: {e}")
        print("Model is saved locally. You can push manually later with:")
        print(f"  huggingface-cli upload {Config.HF_REPO_NAME} {Config.OUTPUT_DIR}/final")
    
    print("\n✅ ALL DONE!")

if __name__ == "__main__":
    main()