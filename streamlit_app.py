"""
Faulty Patch Detection — Streamlit App
=======================================
Run:  streamlit run streamlit_app.py
"""

import os
import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
import streamlit as st
import time
import pickle
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIG
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GOOD_DIR = os.path.join(BASE_DIR, "data", "reference")
BAD_DIR = os.path.join(BASE_DIR, "data", "defective")
DINOV2_CACHE = os.path.join(BASE_DIR, "hexa_sun_model.pkl")
OPENCV_CACHE = os.path.join(BASE_DIR, "hexa_sun_opencv_model.pkl")
VALID_EXT = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
IMG_SIZE = 224


# ============================================================
# DINOV2 DETECTOR
# ============================================================
@st.cache_resource
def load_backbone():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
    backbone = AutoModel.from_pretrained("facebook/dinov2-small")
    backbone = backbone.to(device)
    backbone.eval()
    return processor, backbone, device


@st.cache_resource
def load_dinov2_model():
    if os.path.exists(DINOV2_CACHE):
        with open(DINOV2_CACHE, 'rb') as f:
            cache = pickle.load(f)
        scaler = StandardScaler()
        scaler.mean_ = cache['scaler_mean']
        scaler.scale_ = cache['scaler_scale']
        scaler.n_features_in_ = len(cache['scaler_mean'])
        knn = KNeighborsClassifier(
            n_neighbors=min(cache['best_k'], len(cache['features']) - 1),
            weights='distance')
        knn.fit(scaler.transform(cache['features']), cache['labels'])
        return knn, scaler, cache['loo_accuracy']
    else:
        return build_dinov2_model()


def extract_dinov2_features(pil_image):
    processor, backbone, device = load_backbone()
    inputs = processor(images=pil_image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = backbone(**inputs)
    return outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()


def build_dinov2_model():
    paths, labels, names = _load_dataset_paths()
    features = []
    progress = st.progress(0, text="Extracting DINOv2 features...")
    for i, path in enumerate(paths):
        img = Image.open(path).convert("RGB")
        features.append(extract_dinov2_features(img))
        progress.progress((i + 1) / len(paths), text=f"DINOv2 features... {i+1}/{len(paths)}")
    progress.empty()

    X, y = np.array(features), np.array(labels)
    best_acc, best_k = _find_best_k(X, y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    knn = KNeighborsClassifier(n_neighbors=min(best_k, len(X) - 1), weights='distance')
    knn.fit(X_scaled, y)

    cache = {
        'features': X, 'labels': y, 'names': names,
        'best_k': best_k, 'loo_accuracy': best_acc,
        'scaler_mean': scaler.mean_, 'scaler_scale': scaler.scale_,
    }
    with open(DINOV2_CACHE, 'wb') as f:
        pickle.dump(cache, f)
    return knn, scaler, best_acc


def predict_dinov2(pil_image):
    knn, scaler, _ = load_dinov2_model()
    start = time.perf_counter()
    feat = extract_dinov2_features(pil_image)
    feat_scaled = scaler.transform(feat.reshape(1, -1))
    pred = knn.predict(feat_scaled)[0]
    proba = knn.predict_proba(feat_scaled)[0]
    total_ms = (time.perf_counter() - start) * 1000
    return ['defective', 'reference'][pred], float(max(proba)), total_ms


# ============================================================
# OPENCV DETECTOR
# ============================================================
def extract_opencv_features(pil_image):
    img = np.array(pil_image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    features = []

    # Color histogram (HSV)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    for ch in range(3):
        hist = cv2.calcHist([hsv], [ch], None, [32],
                            [0, 180] if ch == 0 else [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        features.extend(hist)

    # Edge features (Canny)
    edges = cv2.Canny(gray, 50, 150)
    features.append(np.sum(edges > 0) / edges.size)
    grid = 4
    h, w = edges.shape
    ch, cw = h // grid, w // grid
    for i in range(grid):
        for j in range(grid):
            cell = edges[i*ch:(i+1)*ch, j*cw:(j+1)*cw]
            features.append(np.sum(cell > 0) / cell.size)

    # Texture features
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    features.extend([lap.var(), lap.mean()])
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    features.extend([sobelx.var(), sobely.var(), np.sqrt(sobelx**2 + sobely**2).mean()])
    features.append(np.sum(gray > gray.mean()) / gray.size)
    features.extend([gray.mean(), gray.std(), float(np.median(gray)),
                     float(gray.min()), float(gray.max())])
    for q in [gray[:h//2, :w//2], gray[:h//2, w//2:], gray[h//2:, :w//2], gray[h//2:, w//2:]]:
        features.extend([q.mean(), q.std()])

    # ORB keypoints
    orb = cv2.ORB_create(nfeatures=100)
    kps, descs = orb.detectAndCompute(gray, None)
    features.append(len(kps))
    if len(kps) > 0:
        features.extend([np.mean([k.size for k in kps]), np.std([k.size for k in kps]),
                         np.mean([k.response for k in kps]), np.std([k.response for k in kps])])
        if descs is not None:
            features.extend([descs.mean(), descs.std()])
        else:
            features.extend([0, 0])
    else:
        features.extend([0, 0, 0, 0, 0, 0])

    # GLCM-like
    h_diff = np.abs(gray[:, 1:].astype(float) - gray[:, :-1].astype(float))
    v_diff = np.abs(gray[1:, :].astype(float) - gray[:-1, :].astype(float))
    d_diff = np.abs(gray[1:, 1:].astype(float) - gray[:-1, :-1].astype(float))
    features.extend([h_diff.mean(), h_diff.std(), v_diff.mean(), v_diff.std(),
                     d_diff.mean(), d_diff.std()])
    hist_g = cv2.calcHist([gray], [0], None, [32], [0, 256]).flatten()
    hist_g = hist_g / hist_g.sum()
    features.append(-np.sum(hist_g * np.log2(hist_g + 1e-10)))
    features.append(np.sum(hist_g**2))

    return np.array(features, dtype=np.float64)


@st.cache_resource
def load_opencv_model():
    if os.path.exists(OPENCV_CACHE):
        with open(OPENCV_CACHE, 'rb') as f:
            cache = pickle.load(f)
        scaler = StandardScaler()
        scaler.mean_ = cache['scaler_mean']
        scaler.scale_ = cache['scaler_scale']
        scaler.n_features_in_ = len(cache['scaler_mean'])
        clf = SVC(kernel='rbf', probability=True, C=10, gamma='scale')
        clf.fit(scaler.transform(cache['features']), cache['labels'])
        return clf, scaler, cache['loo_accuracy']
    else:
        return build_opencv_model()


def build_opencv_model():
    paths, labels, names = _load_dataset_paths()
    features = []
    progress = st.progress(0, text="Extracting OpenCV features...")
    for i, path in enumerate(paths):
        img = Image.open(path).convert("RGB")
        features.append(extract_opencv_features(img))
        progress.progress((i + 1) / len(paths), text=f"OpenCV features... {i+1}/{len(paths)}")
    progress.empty()

    X, y = np.array(features), np.array(labels)

    # LOO with SVM
    loo = LeaveOneOut()
    correct = 0
    for tr, te in loo.split(X):
        sc = StandardScaler()
        clf = SVC(kernel='rbf', probability=True, C=10, gamma='scale')
        clf.fit(sc.fit_transform(X[tr]), y[tr])
        correct += (clf.predict(sc.transform(X[te]))[0] == y[te[0]])
    loo_acc = correct / len(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    clf = SVC(kernel='rbf', probability=True, C=10, gamma='scale')
    clf.fit(X_scaled, y)

    cache = {
        'features': X, 'labels': y, 'names': names,
        'loo_accuracy': loo_acc,
        'scaler_mean': scaler.mean_, 'scaler_scale': scaler.scale_,
    }
    with open(OPENCV_CACHE, 'wb') as f:
        pickle.dump(cache, f)
    return clf, scaler, loo_acc


def predict_opencv(pil_image):
    clf, scaler, _ = load_opencv_model()
    start = time.perf_counter()
    feat = extract_opencv_features(pil_image)
    feat_scaled = scaler.transform(feat.reshape(1, -1))
    pred = clf.predict(feat_scaled)[0]
    proba = clf.predict_proba(feat_scaled)[0]
    total_ms = (time.perf_counter() - start) * 1000
    return ['defective', 'reference'][pred], float(max(proba)), total_ms


# ============================================================
# SHARED UTILS
# ============================================================
def _load_dataset_paths():
    paths, labels, names = [], [], []
    for fname in sorted(os.listdir(GOOD_DIR)):
        if fname.lower().endswith(VALID_EXT):
            paths.append(os.path.join(GOOD_DIR, fname))
            labels.append(1)
            names.append(fname)
    for fname in sorted(os.listdir(BAD_DIR)):
        if fname.lower().endswith(VALID_EXT):
            paths.append(os.path.join(BAD_DIR, fname))
            labels.append(0)
            names.append(fname)
    return paths, labels, names


def _find_best_k(X, y):
    best_acc, best_k = 0, 3
    for k in [1, 3, 5, 7]:
        correct = 0
        for tr, te in LeaveOneOut().split(X):
            sc = StandardScaler()
            knn = KNeighborsClassifier(n_neighbors=min(k, len(tr)), weights='distance')
            knn.fit(sc.fit_transform(X[tr]), y[tr])
            correct += (knn.predict(sc.transform(X[te]))[0] == y[te[0]])
        if correct / len(y) > best_acc:
            best_acc = correct / len(y)
            best_k = k
    return best_acc, best_k


# ============================================================
# STREAMLIT UI
# ============================================================
st.set_page_config(
    page_title="Faulty Patch Detection",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main { background-color: #0a0a0a; }
    .stApp { background-color: #0a0a0a; }

    .title-wrap { text-align: center; padding: 16px 0 8px; }
    .title-wrap h1 {
        font-size: 36px; font-weight: 800;
        background: linear-gradient(135deg, #60a5fa, #a78bfa);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 4px;
    }
    .title-wrap p { color: #64748b; font-size: 14px; }

    .stat-bar {
        display: flex; gap: 12px; justify-content: center;
        margin: 12px 0 24px; flex-wrap: wrap;
    }
    .stat-chip {
        background: #111827; border: 1px solid #1e293b;
        padding: 6px 16px; border-radius: 20px;
        font-size: 13px; color: #94a3b8;
    }
    .stat-chip b { color: #e2e8f0; }

    .verdict { font-size: 12px; text-transform: uppercase; letter-spacing: 1.5px; font-weight: 700; margin-bottom: 2px; }
    .verdict.faulty { color: #f87171; }
    .verdict.passed { color: #4ade80; }
    .result-label { margin: 0 0 16px; font-size: 28px; font-weight: 800; }
    .result-label.faulty { color: #ef4444; }
    .result-label.passed { color: #22c55e; }

    .metric-row { display: flex; gap: 32px; margin-top: 12px; }
    .metric-item .val { font-size: 22px; font-weight: 700; color: #e2e8f0; }
    .metric-item .lbl { font-size: 11px; color: #4b5563; text-transform: uppercase; }

    .history-item {
        display: flex; align-items: center; gap: 10px;
        padding: 10px 14px; background: #111827; border: 1px solid #1e293b;
        border-radius: 8px; margin-bottom: 4px; font-size: 13px;
    }
    .history-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
    .history-dot.faulty { background: #ef4444; }
    .history-dot.passed { background: #22c55e; }
    .history-fname { flex: 1; color: #94a3b8; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .history-verdict { font-weight: 700; font-size: 12px; }
    .history-verdict.faulty { color: #f87171; }
    .history-verdict.passed { color: #4ade80; }
    .history-meta { color: #374151; font-size: 12px; }

    .model-badge { display: inline-block; padding: 3px 10px; border-radius: 12px; font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; }
    .model-badge.dinov2 { background: #1e1b4b; color: #a78bfa; border: 1px solid #4c1d95; }
    .model-badge.opencv { background: #14291a; color: #4ade80; border: 1px solid #166534; }
</style>
""", unsafe_allow_html=True)

# Sidebar — model switch
st.sidebar.markdown("### Model Selection")
model_choice = st.sidebar.radio(
    "Choose detection engine:",
    ["DINOv2 (Deep Learning)", "OpenCV (Classical CV)"],
    index=0,
    help="DINOv2 uses a pre-trained neural network. OpenCV uses handcrafted features (histograms, edges, textures)."
)
use_dinov2 = model_choice.startswith("DINOv2")


# Title
st.markdown("""
<div class="title-wrap">
    <h1>Faulty Patch Detection</h1>
    <p>AI-powered defect detection for football patch panels</p>
</div>
""", unsafe_allow_html=True)

# Load selected model
if use_dinov2:
    with st.spinner("Loading DINOv2 model..."):
        load_backbone()
        _, _, loo_accuracy = load_dinov2_model()
    model_name = "DINOv2"
    badge_class = "dinov2"
else:
    with st.spinner("Loading OpenCV model..."):
        _, _, loo_accuracy = load_opencv_model()
    model_name = "OpenCV"
    badge_class = "opencv"

device = "cuda" if torch.cuda.is_available() else "cpu"
total_images = len([f for f in os.listdir(GOOD_DIR) if f.lower().endswith(VALID_EXT)]) + \
               len([f for f in os.listdir(BAD_DIR) if f.lower().endswith(VALID_EXT)])

st.markdown(f"""
<div class="stat-bar">
    <div class="stat-chip"><span class="model-badge {badge_class}">{model_name}</span></div>
    <div class="stat-chip">Accuracy: <b>{loo_accuracy:.0%}</b></div>
    <div class="stat-chip">Dataset: <b>{total_images} images</b></div>
    <div class="stat-chip">Device: <b>{device}</b></div>
</div>
""", unsafe_allow_html=True)

# Initialize history
if 'history' not in st.session_state:
    st.session_state.history = []

# Upload
uploaded_file = st.file_uploader(
    "Upload a patch image",
    type=['jpg', 'jpeg', 'png', 'bmp', 'webp'],
    help="Upload a football patch image to check if it's faulty or OK"
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    if use_dinov2:
        label, confidence, time_ms = predict_dinov2(image)
    else:
        label, confidence, time_ms = predict_opencv(image)

    is_faulty = label == 'defective'

    # Result
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(image, use_container_width=True)
    with col2:
        vc = "faulty" if is_faulty else "passed"
        vt = "Faulty" if is_faulty else "Passed"
        lt = "DEFECTIVE" if is_faulty else "OK"

        st.markdown(f"""
        <div style="padding: 8px 0;">
            <span class="model-badge {badge_class}">{model_name}</span>
            <div class="verdict {vc}" style="margin-top:8px;">{vt}</div>
            <div class="result-label {vc}">{lt}</div>
            <div class="metric-row">
                <div class="metric-item">
                    <div class="val">{confidence:.1%}</div>
                    <div class="lbl">Confidence</div>
                </div>
                <div class="metric-item">
                    <div class="val">{time_ms:.0f}ms</div>
                    <div class="lbl">Inference</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Confidence bar
    bar_color = "#ef4444" if is_faulty else "#22c55e"
    st.markdown(f"""
    <div style="height:6px; background:#1e293b; border-radius:3px; margin: 8px 0 24px; overflow:hidden;">
        <div style="height:100%; width:{confidence*100}%; background:{bar_color}; border-radius:3px;"></div>
    </div>
    """, unsafe_allow_html=True)

    # History
    st.session_state.history.insert(0, {
        'name': uploaded_file.name,
        'label': label,
        'confidence': confidence,
        'time_ms': time_ms,
        'model': model_name
    })
    if len(st.session_state.history) > 30:
        st.session_state.history.pop()

# History
if st.session_state.history:
    st.markdown("#### Detection History")
    for h in st.session_state.history:
        is_f = h['label'] == 'defective'
        dc = "faulty" if is_f else "passed"
        vt = "FAULTY" if is_f else "OK"
        m = h.get('model', '?')
        st.markdown(f"""
        <div class="history-item">
            <div class="history-dot {dc}"></div>
            <div class="history-fname">{h['name']}</div>
            <div class="history-meta">{m}</div>
            <div class="history-verdict {dc}">{vt}</div>
            <div class="history-meta">{h['confidence']:.0%}</div>
            <div class="history-meta">{h['time_ms']:.0f}ms</div>
        </div>
        """, unsafe_allow_html=True)
