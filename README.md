# AI-Generated Media Detector
## Final Year Project — Computer Science

A machine learning system to detect AI-generated images and videos using classical feature engineering and scikit-learn classifiers.

---

## Project Structure

```
ai_detector/
├── src/
│   ├── feature_extractor.py    # DCT, LBP, noise, edge feature extraction
│   └── train_model.py          # Training pipeline (RF, SVM, GBM)
├── models/                     # Saved models (after training)
├── templates/
│   └── index.html              # Web UI
├── app.py                      # Flask web application
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Dataset
Create the following directory structure:
```
dataset/
├── real/       # Real images or videos
└── fake/       # AI-generated images or videos
```

Recommended datasets:
- **Images**: CIFAKE (https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)
- **Videos**: FaceForensics++ (https://github.com/ondyari/FaceForensics)

### 3. Train the Model
```bash
# Train image detection model
python src/train_model.py --data_dir ./dataset/images --media_type image

# Train video detection model
python src/train_model.py --data_dir ./dataset/videos --media_type video

# Quick test with limited samples
python src/train_model.py --data_dir ./dataset/images --media_type image --max_samples 200
```

### 4. Run the Web Application
```bash
python app.py
```
Open http://localhost:5000 in your browser.

---

## Feature Engineering

| Feature | Dimensions | Description |
|---------|-----------|-------------|
| DCT Coefficients | 16 | Frequency-domain artifacts from AI upsampling |
| LBP Histogram | 59 | Micro-texture patterns (uniform LBP, R=1, P=8) |
| Color Statistics | 9 | Mean, std, skew per RGB channel |
| Noise Residuals | 8 | High-freq noise absent in AI images |
| Edge Features | 4 | Sobel + Canny edge density |
| **Total (Image)** | **96** | |
| Temporal (Video) | 12 | Inter-frame difference statistics |
| **Total (Video)** | **204** | 96×2 (mean+std across frames) + 12 |

---

## API

### POST /api/predict
Upload a media file for detection.
```bash
curl -X POST -F "file=@image.jpg" http://localhost:5000/api/predict
```
Response:
```json
{
  "success": true,
  "prediction": "AI-Generated",
  "confidence": 94.2,
  "ai_probability": 94.2,
  "real_probability": 5.8,
  "model": "RandomForest",
  "inference_time_s": 0.3,
  "top_features": { ... }
}
```

### GET /api/status
Returns model load status.

---

## Results (Expected)

| Classifier | Image Accuracy | Video Accuracy | F1 (Image) |
|-----------|---------------|---------------|-----------|
| Random Forest | ~92% | ~89% | ~0.92 |
| SVM (RBF) | ~90% | ~85% | ~0.90 |
| Gradient Boosting | ~91% | ~87% | ~0.91 |

---

## Requirements
See requirements.txt
