#!/bin/bash
set -e

echo "========================================"
echo "SROIE Training Pipeline"
echo "========================================"

# Check for HF_TOKEN
if [ -z "$HF_TOKEN" ]; then
    echo "❌ HF_TOKEN not set!"
    echo "Please set it: export HF_TOKEN=your_token_here"
    exit 1
fi

# Install dependencies
echo "📦 Installing dependencies..."
pip install -q -r requirements.txt

# Run training
echo "🚀 Starting training..."
python main.py

echo "✅ Training complete!"
```

**requirements.txt:**
```
transformers==4.44.0
datasets==2.14.0
seqeval==1.2.2
Pillow==10.1.0
torch>=2.0.0
accelerate==0.24.0
huggingface-hub>=0.20.0
python-dotenv==1.0.0
```

**.env (create this file):**
```
HF_TOKEN=your_huggingface_token_here