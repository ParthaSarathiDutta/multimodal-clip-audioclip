# Multimodal Transformer with CLIP + AudioCLIP

This project demonstrates how to align vision and audio using pretrained models (CLIP and AudioCLIP). The system learns a shared embedding space where similar visual and audio inputs are close together.

## Project Goals
- Learn joint embeddings from paired image-audio data
- Perform cross-modal retrieval: retrieve audio from image, and vice versa
- Explore multimodal scene classification using fused embeddings

## Why This Matters
- Multimodal learning is central to perception models
- Useful for tasks like video understanding, scene grounding, and AR/VR

## Project Structure
```
multimodal-clip-audioclip/
├── data/                       # Image-audio dataset (sample set included)
├── models/                    # Fusion model (CLIP + AudioCLIP)
├── scripts/                   # Training and evaluation scripts
├── utils/                     # Dataloader and audio tools
├── plots/                     # Retrieval heatmaps or metrics
├── config.json                # Hyperparameters
├── requirements.txt           # Python packages
└── README.md
```

## How to Run
```bash
pip install -r requirements.txt
python scripts/train.py
python scripts/eval_retrieval.py
```

## Outputs
- Retrieval accuracy: Top-1, Top-5
- Heatmaps: Audio → Image similarity and Image → Audio similarity

## Extensions
- Use ViT + wav2vec2.0 or BEATs instead of CLIP/AudioCLIP
- Add text modality (CLIP supports it)
- Use this for multimodal event detection in videos

## License
MIT — open to modify and use with attribution.
