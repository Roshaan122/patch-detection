"""
Faulty Patch Detection — Streamlit App
=======================================
Run:  streamlit run streamlit_app.py
"""

import os
import base64
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
LOGO_PATH = os.path.join(BASE_DIR, "automaxion-logo.png")
VALID_EXT = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
IMG_SIZE = 224

USERNAME = "automaxiondeveloper"
PASSWORD = "automaxion2026"


# ============================================================
# AUTH
# ============================================================
def check_login():
    if st.session_state.get('authenticated'):
        return True

    st.set_page_config(page_title="Login - Faulty Patch Detection", page_icon="🔒", layout="centered")

    st.markdown("""
    <style>
        .main { background-color: #0a0a0a; }
        .stApp { background-color: #0a0a0a; }
        .login-wrap { max-width: 400px; margin: 60px auto; text-align: center; }
        .login-wrap h2 { color: #e2e8f0; margin-bottom: 4px; }
        .login-wrap p { color: #64748b; font-size: 14px; margin-bottom: 24px; }
    </style>
    """, unsafe_allow_html=True)

    # Logo
    if os.path.exists(LOGO_PATH):
        logo_b64 = base64.b64encode(open(LOGO_PATH, 'rb').read()).decode()
        st.markdown(f"""
        <div style="text-align:center; margin-top:40px;">
            <img src="data:image/png;base64,{logo_b64}" style="height:60px; margin-bottom:16px;">
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="login-wrap">
        <h2>Faulty Patch Detection</h2>
        <p>Sign in to access the detection system</p>
    </div>
    """, unsafe_allow_html=True)

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign In", use_container_width=True)

        if submitted:
            if username == USERNAME and password == PASSWORD:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Invalid username or password")

    return False


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
# OPENCV DETECTOR (kept for future use)
# ============================================================
def extract_opencv_features(pil_image):
    img = np.array(pil_image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = []
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    for ch in range(3):
        hist = cv2.calcHist([hsv], [ch], None, [32],
                            [0, 180] if ch == 0 else [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        features.extend(hist)
    edges = cv2.Canny(gray, 50, 150)
    features.append(np.sum(edges > 0) / edges.size)
    grid = 4
    h, w = edges.shape
    ch_s, cw_s = h // grid, w // grid
    for i in range(grid):
        for j in range(grid):
            cell = edges[i*ch_s:(i+1)*ch_s, j*cw_s:(j+1)*cw_s]
            features.append(np.sum(cell > 0) / cell.size)
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
        'features': X, 'labels': y, 'names': names, 'loo_accuracy': loo_acc,
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

# Login gate — must pass before anything renders
if not check_login():
    st.stop()

# Past this point, user is authenticated
st.set_page_config(
    page_title="Faulty Patch Detection - Automaxion",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .main { background-color: #0a0a0a; }
    .stApp { background-color: #0a0a0a; }

    .banner { text-align: center; padding: 20px 0 8px; }
    .banner img { height: 50px; margin-bottom: 8px; }

    .title-wrap { text-align: center; padding: 0 0 8px; }
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

    .gallery-card {
        background: #111827; border: 2px solid #1e293b; border-radius: 12px;
        overflow: hidden; text-align: center; padding-bottom: 8px;
    }
    .gallery-card.faulty { border-color: #7f1d1d; }
    .gallery-card.passed { border-color: #14532d; }
    .gallery-card img { width: 100%; aspect-ratio: 1; object-fit: cover; }
    .gallery-verdict { font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; margin-top: 6px; }
    .gallery-verdict.faulty { color: #f87171; }
    .gallery-verdict.passed { color: #4ade80; }
    .gallery-meta { font-size: 10px; color: #4b5563; margin-top: 2px; }
    .gallery-fname { font-size: 10px; color: #64748b; margin-top: 2px; padding: 0 6px;
                      overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
</style>
""", unsafe_allow_html=True)

# Banner with logo
if os.path.exists(LOGO_PATH):
    logo_b64 = base64.b64encode(open(LOGO_PATH, 'rb').read()).decode()
    st.markdown(f"""
    <div class="banner">
        <img src="data:image/png;base64,{logo_b64}" alt="Automaxion">
    </div>
    """, unsafe_allow_html=True)

# Title
st.markdown("""
<div class="title-wrap">
    <h1>Faulty Patch Detection</h1>
    <p>AI-powered defect detection for football patch panels</p>
</div>
""", unsafe_allow_html=True)

# Load DINOv2 (default)
with st.spinner("Loading DINOv2 model..."):
    load_backbone()
    _, _, loo_accuracy = load_dinov2_model()

device = "cuda" if torch.cuda.is_available() else "cpu"
total_images = len([f for f in os.listdir(GOOD_DIR) if f.lower().endswith(VALID_EXT)]) + \
               len([f for f in os.listdir(BAD_DIR) if f.lower().endswith(VALID_EXT)])

st.markdown(f"""
<div class="stat-bar">
    <div class="stat-chip">Model: <b>DINOv2</b></div>
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
    label, confidence, time_ms = predict_dinov2(image)
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
            <div class="verdict {vc}">{vt}</div>
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

    # Save image bytes for gallery thumbnail
    import io
    img_buf = io.BytesIO()
    image.save(img_buf, format='JPEG', quality=60)
    img_b64 = base64.b64encode(img_buf.getvalue()).decode()

    st.session_state.history.insert(0, {
        'name': uploaded_file.name,
        'label': label,
        'confidence': confidence,
        'time_ms': time_ms,
        'thumb': img_b64,
    })
    if len(st.session_state.history) > 24:
        st.session_state.history.pop()

# History gallery
if st.session_state.history:
    st.markdown("#### Detection History")
    cols_per_row = 4
    history = st.session_state.history
    for row_start in range(0, len(history), cols_per_row):
        cols = st.columns(cols_per_row)
        for idx, col in enumerate(cols):
            i = row_start + idx
            if i >= len(history):
                break
            h = history[i]
            is_f = h['label'] == 'defective'
            vc = "faulty" if is_f else "passed"
            vt = "FAULTY" if is_f else "OK"
            with col:
                st.markdown(f"""
                <div class="gallery-card {vc}">
                    <img src="data:image/jpeg;base64,{h['thumb']}">
                    <div class="gallery-verdict {vc}">{vt}</div>
                    <div class="gallery-meta">{h['confidence']:.0%} &middot; {h['time_ms']:.0f}ms</div>
                    <div class="gallery-fname" title="{h['name']}">{h['name']}</div>
                </div>
                """, unsafe_allow_html=True)

# Sidebar — logout
st.sidebar.markdown("---")
if st.sidebar.button("Logout"):
    st.session_state.authenticated = False
    st.session_state.history = []
    st.rerun()
