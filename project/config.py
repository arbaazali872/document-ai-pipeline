from pathlib import Path

class Config:
    # Paths
    DATA_DIR = "./sroie_data"
    OUTPUT_DIR = "./models/layoutlmv3-sroie"
    CACHE_DIR = "./cache"
    
    # Training params (RTX 2000 optimized)
    BATCH_SIZE = 2
    GRADIENT_ACCUMULATION = 4
    LEARNING_RATE = 5e-5
    NUM_EPOCHS = 5
    MAX_TRAIN_SAMPLES = None
    MAX_EVAL_SAMPLES = None
    
    # Logging
    LOGGING_STEPS = 10
    EVAL_STEPS = 100
    SAVE_STEPS = 200
    
    # Model
    MODEL_NAME = "microsoft/layoutlmv3-base"
    MAX_LENGTH = 384
    
    # Safety
    MAX_TRAINING_HOURS = 3
    EARLY_STOPPING_PATIENCE = 3
    
    # Hugging Face
    HF_REPO_NAME = "your-username/layoutlmv3-sroie"  # Change this!
    
    @staticmethod
    def setup_dirs():
        Path(Config.CACHE_DIR).mkdir(exist_ok=True)
        Path(Config.OUTPUT_DIR).mkdir(exist_ok=True)