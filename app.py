import os
import sys
import json
import time
import tempfile
import traceback
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import joblib

sys.path.insert(0, str(Path(__file__).parent / 'src'))
from feature_extractor import (extract_image_features_from_path,
                                extract_video_features,
                                FEATURE_NAMES_IMAGE, FEATURE_NAMES_VIDEO)

app = FastAPI(
    title="AI Media Detector API",
    description="Detects AI-generated images and videos using Machine Learning",
    version="1.0.0"
)

# Allow frontend to call API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS_DIR = Path(__file__).parent / 'models'
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}

# Load models at startup
models = {}
scalers = {}
results_meta = {}

def load_models():
    for mtype in ['image', 'video']:
        mp = MODELS_DIR / f'best_model_{mtype}.joblib'
        sp = MODELS_DIR / f'scaler_{mtype}.joblib'
        rp = MODELS_DIR / f'results_{mtype}.json'
        if mp.exists() and sp.exists():
            try:
                models[mtype] = joblib.load(mp)
                scalers[mtype] = joblib.load(sp)
                if rp.exists():
                    with open(rp) as f:
                        results_meta[mtype] = json.load(f)
                print(f"  [OK] Loaded {mtype} model")
            except Exception as e:
                print(f"  [WARN] Could not load {mtype} model: {e}")

load_models()


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path(__file__).parent / 'templates' / 'index.html'
    return HTMLResponse(content=open(html_path, encoding='utf-8').read())


@app.get("/api/status")
async def status():
    return {
        "models_loaded": list(models.keys()),
        "demo_mode": len(models) == 0,
        "supported_image_formats": list(IMAGE_EXTS),
        "supported_video_formats": list(VIDEO_EXTS),
    }


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    # Validate file type
    ext = Path(file.filename).suffix.lower()
    if ext in IMAGE_EXTS:
        media_type = 'image'
    elif ext in VIDEO_EXTS:
        media_type = 'video'
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    start_time = time.time()
    try:
        # Extract features
        if media_type == 'image':
            features = extract_image_features_from_path(tmp_path)
            feature_names = FEATURE_NAMES_IMAGE
        else:
            features = extract_video_features(tmp_path)
            feature_names = FEATURE_NAMES_VIDEO

        inference_time = round(time.time() - start_time, 3)

        # Predict
        demo_mode = media_type not in models
        if demo_mode:
            np.random.seed(int(abs(features.sum()) * 1000) % (2**31 - 1))
            ai_prob = float(np.random.beta(3, 2))
            label = 'AI-Generated' if ai_prob > 0.5 else 'Real'
            confidence = ai_prob if ai_prob > 0.5 else 1 - ai_prob
            model_name = 'Demo'
        else:
            model = models[media_type]
            scaler = scalers[media_type]
            X = scaler.transform(features.reshape(1, -1))
            prob = model.predict_proba(X)[0]
            ai_prob = float(prob[1])
            label = 'AI-Generated' if ai_prob > 0.5 else 'Real'
            confidence = float(max(prob))
            model_name = results_meta.get(media_type, {}).get('best_model', 'Unknown')

        # Top features
        top_features = {}
        if not demo_mode and hasattr(models[media_type], 'feature_importances_'):
            imps = models[media_type].feature_importances_
            top_idx = np.argsort(imps)[::-1][:8]
            top_features = {
                feature_names[i]: {
                    'importance': round(float(imps[i]), 4),
                    'value': round(float(features[i]), 4)
                } for i in top_idx
            }

        return {
            "success": True,
            "filename": file.filename,
            "media_type": media_type,
            "prediction": label,
            "confidence": round(confidence * 100, 1),
            "ai_probability": round(ai_prob * 100, 1),
            "real_probability": round((1 - ai_prob) * 100, 1),
            "model": model_name,
            "demo_mode": demo_mode,
            "inference_time_s": inference_time,
            "feature_count": len(features),
            "top_features": top_features,
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*50)
    print("  AI Media Detector — FastAPI")
    print("="*50)
    print(f"  Models loaded: {list(models.keys()) or 'None (demo mode)'}")
    print(f"  API Docs: http://localhost:7860/docs")
    print(f"  URL:      http://localhost:7860")
    print("="*50 + "\n")
    uvicorn.run("app:app", host="0.0.0.0", port=7860)