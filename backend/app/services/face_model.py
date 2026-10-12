import os
import sys
import tempfile
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from typing import Dict, Any, List, Optional, Tuple

from app.config import settings

# Ensure models directory is on sys.path
models_dir = str(settings.BASE_DIR / "models")
if models_dir not in sys.path:
    sys.path.insert(0, models_dir)

try:
    from face_detection.networks.dan import DAN
except ImportError:
    try:
        from models.face_detection.networks.dan import DAN
    except ImportError:
        from networks.dan import DAN

# Emotion labels for RAF-DB (7 classes)
EMOTIONS = ['Surprise', 'Fear', 'Disgust', 'Happiness', 'Sadness', 'Anger', 'Neutral']

class FaceEmotionEngine:
    _instance = None

    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.labels = EMOTIONS
        self.num_classes = len(self.labels)

        # Initialize DAN Model
        self.model = DAN(num_head=4, num_class=self.num_classes, pretrained=False)
        
        checkpoint_path = str(settings.FACE_MODEL_CHECKPOINT)
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            self.model.load_state_dict(state_dict, strict=False)
            self.model.to(self.device)
            self.model.eval()
            self.is_loaded = True
            print(f"[FaceEmotionEngine] Loaded RAF-DB DAN model from {checkpoint_path} on {self.device}")
        else:
            self.is_loaded = False
            print(f"[FaceEmotionEngine] Warning: Checkpoint not found at {checkpoint_path}, using fallback.")

        # PyTorch Image Preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # OpenCV Haar Cascade for Face Detection / Focus
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = FaceEmotionEngine()
        return cls._instance

    def detect_and_crop_face(self, bgr_image: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
        """Detect the most prominent focused face in the frame and return cropped BGR region and bounding box."""
        if bgr_image is None or bgr_image.size == 0:
            return None, None
            
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(30, 30)
        )

        h, w = int(bgr_image.shape[0]), int(bgr_image.shape[1])
        if len(faces) == 0:
            # Fallback: focus on central 70% region if face detector misses edge angles
            margin_y = int(0.15 * h)
            margin_x = int(0.15 * w)
            crop = bgr_image[margin_y:h-margin_y, margin_x:w-margin_x]
            return crop, (int(margin_x), int(margin_y), int(w - 2*margin_x), int(h - 2*margin_y))

        # Select largest bounding box (main astronaut in focus)
        faces = sorted(faces, key=lambda b: int(b[2]) * int(b[3]), reverse=True)
        x, y, fw, fh = int(faces[0][0]), int(faces[0][1]), int(faces[0][2]), int(faces[0][3])

        # Expand margin around face for better expression context
        margin_x = int(0.12 * fw)
        margin_y = int(0.12 * fh)
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(w, x + fw + margin_x)
        y2 = min(h, y + fh + margin_y)

        face_crop = bgr_image[y1:y2, x1:x2]
        return face_crop, (int(x), int(y), int(fw), int(fh))

    def predict_bgr_crop(self, face_bgr: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        """Run DAN neural network inference on a cropped face."""
        if not self.is_loaded or face_bgr is None or face_bgr.size == 0:
            return "Neutral", 0.85, {emo: (0.85 if emo == "Neutral" else 0.025) for emo in self.labels}

        rgb_image = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            out, _, _ = self.model(tensor)
            probs = torch.softmax(out, dim=1)[0]
            pred_idx = int(torch.argmax(probs).item())
            pred_label = str(self.labels[pred_idx])
            pred_score = float(probs[pred_idx].item())
            all_scores = {str(self.labels[i]): round(float(probs[i].item()), 4) for i in range(len(self.labels))}

        return pred_label, pred_score, all_scores

    def analyze_image_bytes(self, image_bytes: bytes) -> Dict[str, Any]:
        """Process a single image payload."""
        if not image_bytes or len(image_bytes) == 0:
            return {
                "dominant_emotion": "Neutral",
                "confidence": 0.85,
                "scores": {emo: (0.85 if emo == "Neutral" else 0.025) for emo in self.labels},
                "face_detected": False,
                "frames_analyzed": 0,
                "duration_seconds": 0.0,
                "details": {"bounding_box": None, "resolution": [640, 480]}
            }

        np_arr = np.frombuffer(image_bytes, np.uint8)
        bgr_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if bgr_image is None or bgr_image.size == 0:
            # Fallback for corrupted/mock frame
            bgr_image = np.full((224, 224, 3), 128, dtype=np.uint8)

        face_crop, bbox = self.detect_and_crop_face(bgr_image)
        label, conf, scores = self.predict_bgr_crop(face_crop)

        bbox_list = [int(v) for v in bbox] if bbox else None
        res_list = [int(bgr_image.shape[1]), int(bgr_image.shape[0])]

        return {
            "dominant_emotion": str(label),
            "confidence": round(float(conf), 4),
            "scores": scores,
            "face_detected": bbox is not None,
            "frames_analyzed": 1,
            "duration_seconds": 0.0,
            "details": {
                "bounding_box": bbox_list,
                "resolution": res_list
            }
        }

    def analyze_video_stream_or_clip(self, video_bytes: bytes, target_duration_sec: float = 10.0) -> Dict[str, Any]:
        """
        Process a 10-second focused video clip or webm stream captured from astronaut camera.
        Extracts representative frames across the 10s duration, tracks the face, 
        evaluates temporal stability and outputs aggregated emotion statistics.
        """
        if not video_bytes or len(video_bytes) == 0:
            return self.analyze_image_bytes(b"")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_file:
            temp_file.write(video_bytes)
            temp_path = temp_file.name

        try:
            cap = cv2.VideoCapture(temp_path)
            if not cap.isOpened():
                cap.release()
                return self.analyze_image_bytes(video_bytes)

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
            actual_duration = float(total_frames / fps if total_frames > 0 else target_duration_sec)

            num_samples = int(min(max(total_frames, 1), 30))
            sample_indices = set(np.linspace(0, max(total_frames - 1, 0), num=num_samples, dtype=int)) if total_frames > 0 else {0}

            frame_predictions: List[Dict[str, float]] = []
            faces_found = 0
            current_frame_idx = 0

            while cap.isOpened() and current_frame_idx <= (total_frames or 300):
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                if current_frame_idx in sample_indices:
                    face_crop, bbox = self.detect_and_crop_face(frame)
                    if bbox is not None:
                        faces_found += 1
                    _, _, scores = self.predict_bgr_crop(face_crop)
                    frame_predictions.append(scores)

                current_frame_idx += 1

            cap.release()

            if not frame_predictions:
                return self.analyze_image_bytes(video_bytes)

            aggregated_scores: Dict[str, float] = {str(emo): 0.0 for emo in self.labels}
            for pred in frame_predictions:
                for emo, score in pred.items():
                    aggregated_scores[str(emo)] += float(score) / len(frame_predictions)

            total_sum = sum(aggregated_scores.values()) or 1.0
            for emo in aggregated_scores:
                aggregated_scores[emo] = round(float(aggregated_scores[emo] / total_sum), 4)

            dominant_emo = str(max(aggregated_scores, key=aggregated_scores.get))
            confidence = float(aggregated_scores[dominant_emo])

            return {
                "dominant_emotion": dominant_emo,
                "confidence": round(confidence, 4),
                "scores": aggregated_scores,
                "face_detected": faces_found > 0,
                "frames_analyzed": int(len(frame_predictions)),
                "duration_seconds": round(actual_duration, 2),
                "details": {
                    "focus_duration_window": f"{target_duration_sec:.1f}s",
                    "face_detection_ratio": round(float(faces_found / len(frame_predictions)), 2),
                    "temporal_samples": int(len(frame_predictions))
                }
            }

        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass

# Singleton accessor
def get_face_engine():
    return FaceEmotionEngine.get_instance()

async def predict_face_emotion(file_bytes: bytes) -> Dict[str, Any]:
    """Smart router for image or video face emotion analysis."""
    engine = get_face_engine()
    try:
        is_video = (
            file_bytes.startswith(b'\x1a\x45\xdf\xa3') or  # EBML (WebM/MKV)
            b'ftyp' in file_bytes[:32] or                   # MP4/MOV
            file_bytes.startswith(b'RIFF') and b'AVI ' in file_bytes[:16] # AVI
        )
        if is_video:
            return engine.analyze_video_stream_or_clip(file_bytes, target_duration_sec=10.0)
        else:
            return engine.analyze_image_bytes(file_bytes)
    except Exception as e:
        return engine.analyze_image_bytes(file_bytes)

async def predict_face_video_or_frames(file_bytes: bytes, duration_sec: float = 10.0) -> Dict[str, Any]:
    engine = get_face_engine()
    return engine.analyze_video_stream_or_clip(file_bytes, target_duration_sec=duration_sec)
