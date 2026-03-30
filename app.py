"""
Hexa Sun Patch Defect Detector — DINOv2 + Web UI
==================================================
Run:  python3 app.py
Open: http://localhost:5000
"""

import os
import sys
import io
import json
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse
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
MODEL_CACHE = os.path.join(BASE_DIR, "hexa_sun_model.pkl")
VALID_EXT = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
PORT = 8080


class HexaSunDetector:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = None
        self.backbone = None
        self.knn = None
        self.scaler = None
        self.class_names = ['defective', 'reference']
        self.loo_accuracy = 0

    def _load_backbone(self):
        if self.backbone is not None:
            return
        print("Loading DINOv2-small...")
        self.processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
        self.backbone = AutoModel.from_pretrained("facebook/dinov2-small")
        self.backbone = self.backbone.to(self.device)
        self.backbone.eval()
        print(f"DINOv2 loaded on {self.device}")

    def _extract_features_from_pil(self, pil_image):
        self._load_backbone()
        inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.backbone(**inputs)
        return outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()

    def _extract_features(self, image_path):
        image = Image.open(image_path).convert("RGB")
        return self._extract_features_from_pil(image)

    def build_model(self):
        self._load_backbone()
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

        print(f"Extracting features from {len(paths)} images...")
        features = []
        for i, path in enumerate(paths):
            features.append(self._extract_features(path))
            print(f"  [{i+1}/{len(paths)}] {names[i]}")

        X = np.array(features)
        y = np.array(labels)

        best_acc, best_k = 0, 3
        for k in [1, 3, 5, 7]:
            loo = LeaveOneOut()
            correct = 0
            for tr, te in loo.split(X):
                sc = StandardScaler()
                knn = KNeighborsClassifier(n_neighbors=min(k, len(tr)), weights='distance')
                knn.fit(sc.fit_transform(X[tr]), y[tr])
                correct += (knn.predict(sc.transform(X[te]))[0] == y[te[0]])
            acc = correct / len(y)
            if acc > best_acc:
                best_acc = acc
                best_k = k

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        self.knn = KNeighborsClassifier(n_neighbors=min(best_k, len(X) - 1), weights='distance')
        self.knn.fit(X_scaled, y)
        self.loo_accuracy = best_acc

        cache = {
            'features': X, 'labels': y, 'names': names,
            'best_k': best_k, 'loo_accuracy': best_acc,
            'scaler_mean': self.scaler.mean_,
            'scaler_scale': self.scaler.scale_,
        }
        with open(MODEL_CACHE, 'wb') as f:
            pickle.dump(cache, f)
        print(f"Model built! LOO Accuracy: {best_acc:.0%} (k={best_k})")
        return best_acc

    def load_model(self):
        if self.knn is not None:
            return
        if os.path.exists(MODEL_CACHE):
            with open(MODEL_CACHE, 'rb') as f:
                cache = pickle.load(f)
            self.scaler = StandardScaler()
            self.scaler.mean_ = cache['scaler_mean']
            self.scaler.scale_ = cache['scaler_scale']
            self.scaler.n_features_in_ = len(cache['scaler_mean'])
            self.knn = KNeighborsClassifier(
                n_neighbors=min(cache['best_k'], len(cache['features']) - 1),
                weights='distance')
            self.knn.fit(self.scaler.transform(cache['features']), cache['labels'])
            self.loo_accuracy = cache['loo_accuracy']
            print(f"Model loaded (LOO accuracy: {cache['loo_accuracy']:.0%})")
        else:
            self.build_model()

    def predict_pil(self, pil_image):
        self._load_backbone()
        self.load_model()
        start = time.perf_counter()
        feat = self._extract_features_from_pil(pil_image)
        feat_scaled = self.scaler.transform(feat.reshape(1, -1))
        pred = self.knn.predict(feat_scaled)[0]
        proba = self.knn.predict_proba(feat_scaled)[0]
        total_ms = (time.perf_counter() - start) * 1000
        return self.class_names[pred], float(max(proba)), total_ms


# ============================================================
# HTML FRONTEND
# ============================================================
HTML_PAGE = '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Faulty Patch Detection</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0a0a0a; color: #e2e8f0; min-height: 100vh; }

.header { background: linear-gradient(135deg, #0f172a, #1a1a2e); padding: 32px 40px; border-bottom: 1px solid #1e293b; text-align: center; }
.header h1 { font-size: 32px; font-weight: 800; letter-spacing: -0.5px; background: linear-gradient(135deg, #60a5fa, #a78bfa); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.header p { color: #64748b; margin-top: 6px; font-size: 14px; }
.stats { display: flex; gap: 16px; justify-content: center; margin-top: 16px; }
.stat { background: #111827; padding: 8px 20px; border-radius: 24px; font-size: 13px; color: #94a3b8; border: 1px solid #1e293b; }
.stat span { color: #e2e8f0; font-weight: 700; }

.container { max-width: 720px; margin: 0 auto; padding: 40px 24px; }

.upload-zone { border: 2px dashed #2d3748; border-radius: 20px; padding: 64px 32px; text-align: center; cursor: pointer; transition: all 0.3s; background: #111827; position: relative; overflow: hidden; }
.upload-zone::before { content: ''; position: absolute; inset: 0; background: radial-gradient(circle at 50% 50%, rgba(59,130,246,0.05), transparent 70%); pointer-events: none; }
.upload-zone:hover, .upload-zone.dragover { border-color: #3b82f6; background: #0f1a2e; }
.upload-zone .icon { font-size: 48px; margin-bottom: 16px; opacity: 0.6; }
.upload-zone h2 { font-size: 18px; margin-bottom: 6px; font-weight: 600; }
.upload-zone p { color: #4b5563; font-size: 13px; }
.upload-zone input { display: none; }

.result-card { background: #111827; border-radius: 20px; padding: 0; margin: 32px 0; display: none; border: 2px solid #1e293b; overflow: hidden; }
.result-card.show { display: block; animation: fadeIn 0.4s ease; }
.result-card.defective { border-color: #dc2626; }
.result-card.reference { border-color: #16a34a; }
@keyframes fadeIn { from { opacity: 0; transform: translateY(12px); } to { opacity: 1; transform: translateY(0); } }

.result-top { display: flex; align-items: center; gap: 20px; padding: 24px; }
.result-img { width: 140px; height: 140px; object-fit: cover; border-radius: 14px; flex-shrink: 0; border: 2px solid #1e293b; }
.result-info { flex: 1; }
.result-verdict { font-size: 13px; text-transform: uppercase; letter-spacing: 1.5px; font-weight: 600; margin-bottom: 4px; }
.result-verdict.defective { color: #f87171; }
.result-verdict.reference { color: #4ade80; }
.result-label { font-size: 32px; font-weight: 800; margin-bottom: 12px; }
.result-label.defective { color: #ef4444; }
.result-label.reference { color: #22c55e; }

.result-stats { display: flex; gap: 24px; }
.result-stat { text-align: left; }
.result-stat .val { font-size: 20px; font-weight: 700; color: #e2e8f0; }
.result-stat .lbl { font-size: 11px; color: #4b5563; text-transform: uppercase; letter-spacing: 0.5px; margin-top: 2px; }

.confidence-bar-wrap { padding: 0 24px 24px; }
.confidence-bar { height: 6px; background: #1e293b; border-radius: 3px; overflow: hidden; }
.confidence-fill { height: 100%; border-radius: 3px; transition: width 0.6s ease; }

.loading { display: none; text-align: center; padding: 40px; }
.loading.show { display: block; }
.spinner { width: 36px; height: 36px; border: 3px solid #1e293b; border-top-color: #3b82f6; border-radius: 50%; animation: spin 0.7s linear infinite; margin: 0 auto 16px; }
@keyframes spin { to { transform: rotate(360deg); } }
.loading p { color: #4b5563; font-size: 14px; }

.history { margin-top: 40px; }
.history-title { font-size: 14px; font-weight: 600; color: #4b5563; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 12px; }
.history-item { display: flex; align-items: center; gap: 12px; padding: 12px 16px; background: #111827; border-radius: 10px; margin-bottom: 4px; font-size: 13px; border: 1px solid #1e293b; transition: background 0.2s; }
.history-item:hover { background: #1a202c; }
.history-item .dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
.history-item .dot.defective { background: #ef4444; }
.history-item .dot.reference { background: #22c55e; }
.history-item .fname { flex: 1; color: #94a3b8; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.history-item .verdict { font-weight: 700; font-size: 12px; }
.history-item .verdict.defective { color: #f87171; }
.history-item .verdict.reference { color: #4ade80; }
.history-item .meta { color: #374151; font-size: 12px; }

@media (max-width: 640px) {
    .result-top { flex-direction: column; text-align: center; }
    .result-img { width: 100%; height: 200px; }
    .result-stats { justify-content: center; }
    .stats { flex-direction: column; align-items: center; }
}
</style>
</head>
<body>

<div class="header">
    <h1>Faulty Patch Detection</h1>
    <p>AI-powered defect detection for football patch panels</p>
    <div class="stats">
        <div class="stat" id="modelStat">Loading model...</div>
    </div>
</div>

<div class="container">
    <div class="upload-zone" id="uploadZone">
        <div class="icon">+</div>
        <h2>Upload a patch image</h2>
        <p>Drop an image here or click to browse &mdash; JPG, PNG, BMP, WEBP</p>
        <input type="file" id="fileInput" accept="image/*">
    </div>

    <div class="loading" id="loading">
        <div class="spinner"></div>
        <p>Analyzing patch...</p>
    </div>

    <div class="result-card" id="resultCard">
        <div class="result-top">
            <img class="result-img" id="resultImg">
            <div class="result-info">
                <div class="result-verdict" id="resultVerdict"></div>
                <div class="result-label" id="resultLabel"></div>
                <div class="result-stats">
                    <div class="result-stat">
                        <div class="val" id="resultConf"></div>
                        <div class="lbl">Confidence</div>
                    </div>
                    <div class="result-stat">
                        <div class="val" id="resultTime"></div>
                        <div class="lbl">Inference</div>
                    </div>
                    <div class="result-stat">
                        <div class="val" id="resultFile"></div>
                        <div class="lbl">File</div>
                    </div>
                </div>
            </div>
        </div>
        <div class="confidence-bar-wrap">
            <div class="confidence-bar">
                <div class="confidence-fill" id="confFill"></div>
            </div>
        </div>
    </div>

    <div class="history" id="history" style="display:none">
        <div class="history-title">Detection History</div>
        <div id="historyList"></div>
    </div>
</div>

<script>
const uploadZone = document.getElementById('uploadZone');
const fileInput = document.getElementById('fileInput');
const loading = document.getElementById('loading');
const resultCard = document.getElementById('resultCard');
const historySection = document.getElementById('history');
const historyList = document.getElementById('historyList');
let history = [];

fetch('/api/status').then(r => r.json()).then(data => {
    document.getElementById('modelStat').innerHTML =
        'Model Accuracy: <span>' + (data.loo_accuracy * 100).toFixed(0) + '%</span> &nbsp; | &nbsp; ' +
        'Dataset: <span>' + data.total_images + ' images</span> &nbsp; | &nbsp; ' +
        'Device: <span>' + data.device + '</span>';
});

uploadZone.onclick = () => fileInput.click();
fileInput.onchange = (e) => { if (e.target.files[0]) uploadAndPredict(e.target.files[0]); };
uploadZone.ondragover = (e) => { e.preventDefault(); uploadZone.classList.add('dragover'); };
uploadZone.ondragleave = () => uploadZone.classList.remove('dragover');
uploadZone.ondrop = (e) => {
    e.preventDefault();
    uploadZone.classList.remove('dragover');
    if (e.dataTransfer.files[0]) uploadAndPredict(e.dataTransfer.files[0]);
};

function uploadAndPredict(file) {
    loading.classList.add('show');
    resultCard.classList.remove('show');
    const formData = new FormData();
    formData.append('image', file);
    fetch('/api/predict', { method: 'POST', body: formData })
        .then(r => r.json())
        .then(data => showResult(data, URL.createObjectURL(file), file.name))
        .catch(err => { loading.classList.remove('show'); alert('Error: ' + err); });
}

function showResult(data, imgSrc, fileName) {
    loading.classList.remove('show');
    const isFaulty = data.label === 'defective';
    resultCard.className = 'result-card show ' + data.label;
    document.getElementById('resultImg').src = imgSrc;
    document.getElementById('resultVerdict').textContent = isFaulty ? 'Faulty' : 'Passed';
    document.getElementById('resultVerdict').className = 'result-verdict ' + data.label;
    document.getElementById('resultLabel').textContent = isFaulty ? 'DEFECTIVE' : 'OK';
    document.getElementById('resultLabel').className = 'result-label ' + data.label;
    document.getElementById('resultConf').textContent = (data.confidence * 100).toFixed(1) + '%';
    document.getElementById('resultTime').textContent = data.time_ms.toFixed(0) + 'ms';
    const shortName = fileName.length > 18 ? fileName.substring(0, 15) + '...' : fileName;
    document.getElementById('resultFile').textContent = shortName;

    const fill = document.getElementById('confFill');
    fill.style.width = (data.confidence * 100) + '%';
    fill.style.background = isFaulty ? '#ef4444' : '#22c55e';

    history.unshift({ label: data.label, confidence: data.confidence, name: fileName, time: data.time_ms });
    if (history.length > 30) history.pop();
    renderHistory();

    resultCard.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

function renderHistory() {
    if (history.length === 0) return;
    historySection.style.display = 'block';
    historyList.innerHTML = history.map(h => {
        const isFaulty = h.label === 'defective';
        return `<div class="history-item">
            <div class="dot ${h.label}"></div>
            <div class="fname">${h.name}</div>
            <div class="verdict ${h.label}">${isFaulty ? 'FAULTY' : 'OK'}</div>
            <div class="meta">${(h.confidence*100).toFixed(0)}%</div>
            <div class="meta">${h.time.toFixed(0)}ms</div>
        </div>`;
    }).join('');
}
</script>
</body>
</html>'''


# ============================================================
# HTTP SERVER
# ============================================================
detector = HexaSunDetector()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # suppress request logs

    def _send_json(self, data, status=200):
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def _send_html(self, html):
        self.send_response(200)
        self.send_header('Content-Type', 'text/html')
        self.end_headers()
        self.wfile.write(html.encode())

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == '/' or parsed.path == '':
            self._send_html(HTML_PAGE)

        elif parsed.path == '/api/status':
            detector._load_backbone()
            detector.load_model()
            self._send_json({
                'loo_accuracy': detector.loo_accuracy,
                'total_images': len(os.listdir(GOOD_DIR)) + len(os.listdir(BAD_DIR)),
                'device': str(detector.device),
            })

        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == '/api/predict':
            content_length = int(self.headers['Content-Length'])
            body = self.rfile.read(content_length)

            # Parse multipart form data
            boundary = self.headers['Content-Type'].split('boundary=')[1].encode()
            parts = body.split(b'--' + boundary)
            image_data = None
            filename = 'uploaded.jpg'

            for part in parts:
                if b'filename=' in part:
                    header_end = part.index(b'\r\n\r\n') + 4
                    image_data = part[header_end:].rstrip(b'\r\n--')
                    fname_start = part.index(b'filename="') + 10
                    fname_end = part.index(b'"', fname_start)
                    filename = part[fname_start:fname_end].decode()

            if image_data:
                pil_image = Image.open(io.BytesIO(image_data)).convert("RGB")
                label, confidence, time_ms = detector.predict_pil(pil_image)
                self._send_json({
                    'label': label, 'confidence': confidence,
                    'time_ms': time_ms, 'filename': filename
                })
            else:
                self._send_json({'error': 'No image found'}, 400)

        else:
            self.send_response(404)
            self.end_headers()


def main():
    if len(sys.argv) > 1 and sys.argv[1] == '--build':
        detector.build_model()
        return

    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        detector._load_backbone()
        print("\nLOO Test:")
        paths, labels = [], []
        for f in sorted(os.listdir(GOOD_DIR)):
            if f.lower().endswith(VALID_EXT):
                paths.append(os.path.join(GOOD_DIR, f))
                labels.append(1)
        for f in sorted(os.listdir(BAD_DIR)):
            if f.lower().endswith(VALID_EXT):
                paths.append(os.path.join(BAD_DIR, f))
                labels.append(0)
        features = [detector._extract_features(p) for p in paths]
        X, y = np.array(features), np.array(labels)
        for k in [1, 3, 5]:
            correct = 0
            for tr, te in LeaveOneOut().split(X):
                sc = StandardScaler()
                knn = KNeighborsClassifier(n_neighbors=min(k, len(tr)), weights='distance')
                knn.fit(sc.fit_transform(X[tr]), y[tr])
                correct += (knn.predict(sc.transform(X[te]))[0] == y[te[0]])
            print(f"  k={k}: {correct}/{len(y)} = {correct/len(y):.0%}")
        return

    # Start web server
    print(f"\nStarting Hexa Sun Detector...")
    print(f"Loading model...")
    detector._load_backbone()
    detector.load_model()
    print(f"\n{'='*50}")
    print(f"  Open in browser: http://localhost:{PORT}")
    print(f"{'='*50}\n")

    server = HTTPServer(('0.0.0.0', PORT), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
        server.server_close()


if __name__ == "__main__":
    main()
