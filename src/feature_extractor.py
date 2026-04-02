import numpy as np
import cv2
from scipy.fftpack import dct
from skimage.feature import local_binary_pattern
from skimage.color import rgb2gray
from skimage.filters import sobel
import warnings
warnings.filterwarnings('ignore')

IMG_SIZE = 224
LBP_RADIUS = 1
LBP_N_POINTS = 8
LBP_METHOD = 'uniform'


def preprocess_image(img_bgr):
    img = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb.astype(np.float32) / 255.0


def extract_dct_features(img_rgb):
    gray = rgb2gray(img_rgb)
    h, w = gray.shape
    block_size = 8
    coeffs = []
    for i in range(0, h - block_size + 1, block_size):
        for j in range(0, w - block_size + 1, block_size):
            block = gray[i:i+block_size, j:j+block_size]
            d = dct(dct(block.T, norm='ortho').T, norm='ortho')
            coeffs.append(d.flatten())
    if not coeffs:
        return np.zeros(16)
    coeffs = np.array(coeffs)
    bands = [coeffs[:, :4].flatten(),
             coeffs[:, 4:16].flatten(),
             coeffs[:, 16:32].flatten(),
             coeffs[:, 32:].flatten()]
    features = []
    for band in bands:
        features += [np.mean(np.abs(band)),
                     np.var(band),
                     float(np.mean((band - band.mean())**3) / (band.std()**3 + 1e-8)),
                     float(np.mean((band - band.mean())**4) / (band.std()**4 + 1e-8))]
    return np.array(features, dtype=np.float32)


def extract_lbp_features_full(img_rgb):
    gray = (rgb2gray(img_rgb) * 255).astype(np.uint8)
    lbp = local_binary_pattern(gray, 8, 1, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=59, range=(0, 59), density=True)
    return hist.astype(np.float32)


def extract_color_features(img_rgb):
    features = []
    for c in range(3):
        ch = img_rgb[:, :, c].flatten()
        mean = np.mean(ch)
        std = np.std(ch) + 1e-8
        skew = float(np.mean(((ch - mean) / std) ** 3))
        features += [mean, std, skew]
    return np.array(features, dtype=np.float32)


def extract_noise_features(img_rgb):
    gray = rgb2gray(img_rgb).astype(np.float32)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    residual = gray - blurred
    r = residual.flatten()
    std = np.std(r) + 1e-8
    yuv = cv2.cvtColor((img_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2YUV)
    lum = yuv[:, :, 0].astype(np.float32) / 255.0
    blurred_l = cv2.GaussianBlur(lum, (5, 5), 0)
    residual_l = (lum - blurred_l).flatten()
    std_l = np.std(residual_l) + 1e-8
    features = [
        np.mean(np.abs(r)), np.std(r),
        float(np.mean(((r - r.mean()) / std) ** 3)),
        float(np.mean(((r - r.mean()) / std) ** 4)),
        np.mean(np.abs(residual_l)), np.std(residual_l),
        float(np.mean(((residual_l - residual_l.mean()) / std_l) ** 3)),
        float(np.mean(((residual_l - residual_l.mean()) / std_l) ** 4)),
    ]
    return np.array(features, dtype=np.float32)


def extract_edge_features(img_rgb):
    gray = rgb2gray(img_rgb).astype(np.float32)
    sobel_map = sobel(gray)
    sobel_mean = np.mean(sobel_map)
    sobel_std = np.std(sobel_map)
    gray_uint8 = (gray * 255).astype(np.uint8)
    canny = cv2.Canny(gray_uint8, 50, 150)
    edge_density = np.sum(canny > 0) / canny.size
    edge_gradient = np.mean(canny[canny > 0]) if np.any(canny > 0) else 0.0
    return np.array([sobel_mean, sobel_std, edge_density, edge_gradient], dtype=np.float32)


def extract_image_features(img_bgr):
    img_rgb = preprocess_image(img_bgr)
    dct_f   = extract_dct_features(img_rgb)
    lbp_f   = extract_lbp_features_full(img_rgb)
    color_f = extract_color_features(img_rgb)
    noise_f = extract_noise_features(img_rgb)
    edge_f  = extract_edge_features(img_rgb)
    return np.concatenate([dct_f, lbp_f, color_f, noise_f, edge_f])


def extract_image_features_from_path(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    return extract_image_features(img)


def extract_video_features(video_path, fps_sample=1, max_frames=30):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25
    frame_interval = max(1, int(fps / fps_sample))
    frame_features = []
    prev_frame_gray = None
    temporal_diffs = []
    frame_count = 0
    while len(frame_features) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            feats = extract_image_features(frame)
            frame_features.append(feats)
            gray = cv2.cvtColor(cv2.resize(frame, (64, 64)), cv2.COLOR_BGR2GRAY)
            if prev_frame_gray is not None:
                diff = np.abs(gray.astype(float) - prev_frame_gray.astype(float))
                basic = [
                    np.mean(diff), np.std(diff),
                    np.max(diff), np.percentile(diff, 75)
                ]
                flow = cv2.calcOpticalFlowFarneback(
                    prev_frame_gray, gray, None,
                    0.5, 3, 15, 3, 5, 1.2, 0
                )
                flow_mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
                flow_stats = [
                    np.mean(flow_mag), np.std(flow_mag),
                    np.max(flow_mag), np.percentile(flow_mag, 75)
                ]
                consistency = [
                    np.mean(np.abs(diff - np.mean(diff))),
                    np.percentile(diff, 90) - np.percentile(diff, 10),
                ]
                temporal_diffs.append(basic + flow_stats + consistency)
            prev_frame_gray = gray
        frame_count += 1
    cap.release()
    if not frame_features:
        return np.zeros(222)
    frame_features = np.array(frame_features)
    frame_mean = np.mean(frame_features, axis=0)
    frame_std  = np.std(frame_features, axis=0)
    if temporal_diffs:
        temporal_diffs = np.array(temporal_diffs)
        temporal_feat = np.concatenate([
            np.mean(temporal_diffs, axis=0),
            np.std(temporal_diffs, axis=0),
            np.max(temporal_diffs, axis=0),
        ])
    else:
        temporal_feat = np.zeros(30)
    return np.concatenate([frame_mean, frame_std, temporal_feat])


FEATURE_NAMES_IMAGE = (
    [f"dct_band{b}_{s}" for b in range(4) for s in ['mean','var','skew','kurt']] +
    [f"lbp_{i}" for i in range(59)] +
    [f"color_{c}_{s}" for c in ['r','g','b'] for s in ['mean','std','skew']] +
    [f"noise_{i}" for i in range(8)] +
    ['edge_sobel_mean','edge_sobel_std','edge_canny_density','edge_canny_grad']
)

FEATURE_NAMES_VIDEO = (
    [f"frame_mean_{n}" for n in FEATURE_NAMES_IMAGE] +
    [f"frame_std_{n}" for n in FEATURE_NAMES_IMAGE] +
    [f"temporal_{t}_{s}" for t in range(10) for s in ['mean','std','max']]
)
