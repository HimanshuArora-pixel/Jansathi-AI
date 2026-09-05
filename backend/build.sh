#!/usr/bin/env bash
set -o errexit

pip install -r requirements.txt

# Pre-download the quantized ONNX model during build so startup is instant
python -c "
import os, urllib.request
target_dir = 'models/intent_classifier'
os.makedirs(target_dir, exist_ok=True)
target_path = os.path.join(target_dir, 'model_quantized.onnx')
if not os.path.exists(target_path) or os.path.getsize(target_path) < 10000000:
    print('Downloading model_quantized.onnx from Hugging Face during build...')
    urllib.request.urlretrieve('https://huggingface.co/EverVissionAI/jansaathi-legal-intent/resolve/main/model_quantized.onnx', target_path)
    print('Model downloaded successfully! Size:', os.path.getsize(target_path))
else:
    print('Model already present!')
"
