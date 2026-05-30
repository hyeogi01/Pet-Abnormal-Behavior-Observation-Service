import io
import os
import math
import torch
import librosa
import numpy as np
import tempfile
import cv2
import torch.nn as nn
from PIL import Image

def sanitize_data(data):
    """Recursively replace NaN, Inf, -Inf with 0.0 in dicts and lists."""
    if isinstance(data, dict):
        return {k: sanitize_data(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_data(v) for v in data]
    elif isinstance(data, float):
        if math.isnan(data) or math.isinf(data):
            return 0.0
        return data
    return data

# ───────────────────────────────── CONFIG ─────────────────────────────────────
SIGLIP_ID    = "google/siglip-so400m-patch14-384"
WAV2VEC2_ID  = "facebook/wav2vec2-base"
SR           = 16000
MAX_AUDIO_LEN = SR * 5
PATELLA_FEAT_DIM = 59

DOG_BEHAVIOR = ["DOG_BODYLOWER","DOG_BODYSCRATCH","DOG_BODYSHAKE","DOG_FEETUP",
                "DOG_FOOTUP","DOG_HEADING","DOG_LYING","DOG_MOUNTING",
                "DOG_SIT","DOG_TAILING","DOG_TAILLOW","DOG_TURN","DOG_WALKRUN"]
DOG_EMOTION  = ["dog_angry","dog_anxious","dog_confused","dog_happy","dog_relaxed","dog_sad"]
DOG_SOUND    = ["dog_bark","dog_howling","dog_respiratory_event","dog_whining"]
DOG_PATELLA  = ["1","2","3","4","normal"]

CAT_BEHAVIOR = ["CAT_ARCH","CAT_ARMSTRETCH","CAT_FOOTPUSH","CAT_GETDOWN",
                "CAT_GROOMING","CAT_HEADING","CAT_LAYDOWN","CAT_LYING",
                "CAT_ROLL","CAT_SITDOWN","CAT_TAILING","CAT_WALKRUN"]
CAT_EMOTION  = ["cat_relaxed","cat_happy","cat_attentive","cat_sad"]
CAT_SOUND    = ["cat_aggressive","cat_positive"]

# ─────────────────────────────── MODEL ────────────────────────────────────────
class ClassificationHeads(nn.Module):
    def __init__(self, img_dim, aud_dim):
        super().__init__()
        self.img_dim = img_dim
        self.aud_dim = aud_dim

        def _head(in_dim, n_cls):
            return nn.Sequential(
                nn.Dropout(0.3), nn.Linear(in_dim, 256),
                nn.BatchNorm1d(256), nn.GELU(),
                nn.Dropout(0.2), nn.Linear(256, n_cls),
            )

        self.dog_behavior = _head(img_dim, len(DOG_BEHAVIOR))
        self.dog_emotion  = _head(img_dim, len(DOG_EMOTION))
        self.cat_behavior = _head(img_dim, len(CAT_BEHAVIOR))
        self.cat_emotion  = _head(img_dim, len(CAT_EMOTION))

        self.dog_sound = _head(aud_dim, len(DOG_SOUND))
        self.cat_sound = _head(aud_dim, len(CAT_SOUND))

        self.patella_branch = nn.Sequential(
            nn.Linear(PATELLA_FEAT_DIM, 128), nn.LayerNorm(128), nn.GELU(),
            nn.Dropout(0.2), nn.Linear(128, 128), nn.GELU(),
        )
        self.dog_patella = _head(img_dim + 128, len(DOG_PATELLA))

    def forward(self, h, species, task, patella_feat=None):
        if task == "patella" and species == "dog" and patella_feat is not None:
            pf = self.patella_branch(patella_feat)
            return self.dog_patella(torch.cat([h, pf], 1))
        heads = {
            ("dog","behavior"): self.dog_behavior, ("dog","emotion"): self.dog_emotion,
            ("dog","sound"): self.dog_sound,
            ("cat","behavior"): self.cat_behavior, ("cat","emotion"): self.cat_emotion,
            ("cat","sound"): self.cat_sound,
        }
        return heads[(species, task)](h)


# ─────────────────────────────── BACKBONE LOADERS ─────────────────────────────
def _load_siglip(device):
    from transformers import AutoModel, SiglipImageProcessor
    print(f"Loading SigLIP: {SIGLIP_ID}")
    model = AutoModel.from_pretrained(SIGLIP_ID, torch_dtype=torch.float16)
    model = model.vision_model.to(device).eval()
    processor = SiglipImageProcessor.from_pretrained(SIGLIP_ID)
    print(f"  SigLIP loaded. hidden_dim={model.config.hidden_size}")
    return model, processor


def _load_wav2vec2(device):
    from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
    print(f"Loading Wav2Vec2: {WAV2VEC2_ID}")
    model = Wav2Vec2Model.from_pretrained(WAV2VEC2_ID, torch_dtype=torch.float16)
    model = model.to(device).eval()
    processor = Wav2Vec2FeatureExtractor.from_pretrained(WAV2VEC2_ID)
    print(f"  Wav2Vec2 loaded. hidden_dim={model.config.hidden_size}")
    return model, processor


# ─────────────────────────────── ENGINE ───────────────────────────────────────
class DailyBehaviorEngine:
    def __init__(self):
        self.is_loaded = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.dog_behavior_classes = []
        self.dog_emotion_classes  = []
        self.dog_sound_classes    = []
        self.dog_patella_classes  = []
        self.cat_behavior_classes = []
        self.cat_emotion_classes  = []
        self.cat_sound_classes    = []

        self.siglip_model    = None
        self.siglip_processor = None
        self.wav2vec2_model  = None
        self.feature_extractor = None
        self.heads           = None

    def _safe_float(self, val):
        try:
            f = float(val)
            if math.isnan(f) or math.isinf(f):
                return 0.0
            return f
        except:
            return 0.0

    def load_models(self):
        print("Loading Daily Behavior Omni Models (SigLIP + Wav2Vec2 + Heads)...")
        base_dir  = "/app/AI_pth" if os.path.exists("/app/AI_pth") else "AI Model/AI_pth"
        ckpt_path = os.path.join(base_dir, "normal_omni_best.pth")

        if not os.path.exists(ckpt_path):
            print(f"Checkpoint not found: {ckpt_path}")
            return

        try:
            self.siglip_model, self.siglip_processor = _load_siglip(self.device)
            self.wav2vec2_model, self.feature_extractor = _load_wav2vec2(self.device)

            ckpt    = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            img_dim = ckpt["img_dim"]
            aud_dim = ckpt["aud_dim"]
            classes = ckpt["classes"]

            self.heads = ClassificationHeads(img_dim, aud_dim).to(self.device)
            self.heads.load_state_dict(ckpt["heads_state_dict"])
            self.heads.eval()

            self.dog_behavior_classes = classes.get("dog_behavior", DOG_BEHAVIOR)
            self.dog_emotion_classes  = classes.get("dog_emotion",  DOG_EMOTION)
            self.dog_sound_classes    = classes.get("dog_sound",    DOG_SOUND)
            self.dog_patella_classes  = classes.get("dog_patella",  DOG_PATELLA)
            self.cat_behavior_classes = classes.get("cat_behavior", CAT_BEHAVIOR)
            self.cat_emotion_classes  = classes.get("cat_emotion",  CAT_EMOTION)
            self.cat_sound_classes    = classes.get("cat_sound",    CAT_SOUND)

            self.is_loaded = True
            print(f"Models loaded successfully from {ckpt_path}")
            print(f"  img_dim={img_dim}, aud_dim={aud_dim}")
        except Exception as e:
            print(f"Failed to load models: {e}")
            import traceback; traceback.print_exc()

    def extract_frames(self, video_path: str):
        """Extract 1 frame/sec as PIL Images for SigLIP."""
        frames = []
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30
        frame_interval = int(round(fps))
        count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if count % frame_interval == 0:
                frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            count += 1
            if len(frames) >= 20:
                break
        cap.release()
        return frames if frames else None

    def extract_audio_tensor(self, video_path: str):
        if self.feature_extractor is None:
            return None
        try:
            w, _ = librosa.load(video_path, sr=SR, mono=True)
            w = (w[:MAX_AUDIO_LEN] if len(w) > MAX_AUDIO_LEN
                 else np.pad(w, (0, MAX_AUDIO_LEN - len(w))).astype(np.float32))
            inp = self.feature_extractor(w, sampling_rate=SR, return_tensors="pt")
            return inp.input_values.to(self.device)
        except Exception as e:
            print(f"Audio extraction error: {e}")
            return None

    def _siglip_features(self, images_pil):
        """Run SigLIP vision encoder on a list of PIL Images → (N, img_dim)."""
        inputs = self.siglip_processor(images=images_pil, return_tensors="pt")
        pv = inputs["pixel_values"].to(self.device, dtype=torch.float16)
        with torch.no_grad():
            h = self.siglip_model(pixel_values=pv).last_hidden_state.mean(dim=1).float()
        return h

    def _infer_image_task(self, features, species, task, classes):
        if features is None or features.shape[0] == 0:
            return "Unknown", 0.0
        with torch.no_grad():
            logits = self.heads(features, species, task)
            probs  = torch.softmax(logits, dim=1).mean(dim=0)
            max_prob, max_idx = torch.max(probs, dim=0)
            return classes[max_idx.item()], self._safe_float(round(max_prob.item(), 3))

    def _infer_audio_task(self, audio_tensor, species, classes):
        if audio_tensor is None or audio_tensor.shape[0] == 0:
            return "Unknown", 0.0
        try:
            with torch.no_grad():
                h = self.wav2vec2_model(
                    input_values=audio_tensor.half()
                ).last_hidden_state
                if torch.isnan(h).any():
                    return "Unknown", 0.0
                features = h.mean(dim=1).float()
                logits   = self.heads(features, species, "sound")
                probs    = torch.softmax(logits, dim=1)
                max_prob, max_idx = torch.max(probs[0], dim=0)
                return classes[max_idx.item()], self._safe_float(round(max_prob.item(), 4))
        except Exception as e:
            print(f"[AI ENGINE] Audio inference error: {e}")
            return "Unknown", 0.0

    def _infer_patella_task(self, features, classes):
        if features is None or features.shape[0] == 0:
            return "Unknown", 0.0
        feat = torch.zeros(
            (features.shape[0], PATELLA_FEAT_DIM), dtype=torch.float32, device=self.device
        )
        with torch.no_grad():
            logits = self.heads(features, "dog", "patella", patella_feat=feat)
            probs  = torch.softmax(logits, dim=1).mean(dim=0)
            max_prob, max_idx = torch.max(probs, dim=0)
            return classes[max_idx.item()], self._safe_float(round(max_prob.item(), 3))

    def _wait_for_load(self):
        import time
        wait_start = time.time()
        while not self.is_loaded:
            if time.time() - wait_start > 60:
                print("[ERROR] Timeout waiting for AI models to load.")
                break
            print("[WAIT] AI Models are still loading... waiting 2s.")
            time.sleep(2)

    def _normalize_pet_type(self, pet_type: str):
        raw = str(pet_type).lower().strip()
        is_dog = any(k in raw for k in ["dog","doggy","puppy","강아지","개","견"])
        is_cat = any(k in raw for k in ["cat","kitty","고양이","묘"])
        if not (is_dog or is_cat):
            is_dog = True
        return "dog" if is_dog else "cat"

    def analyze_image(self, image_data: bytes, pet_type: str = "dog") -> dict:
        self._wait_for_load()
        if not self.is_loaded:
            return {"status": "error", "message": "Behavior models are not loaded yet."}

        pt_label = self._normalize_pet_type(pet_type)
        print(f"[AI ENGINE] Analyzing image for pet_type: {pet_type} (Mapped to: {pt_label})")

        try:
            img_pil  = Image.open(io.BytesIO(image_data)).convert("RGB")
            features = self._siglip_features([img_pil])  # (1, 1152)

            if pt_label == "dog":
                beh, b_conf = self._infer_image_task(features, "dog", "behavior", self.dog_behavior_classes)
                emo, e_conf = self._infer_image_task(features, "dog", "emotion",  self.dog_emotion_classes)
                pat, p_conf = self._infer_patella_task(features, self.dog_patella_classes)
                res_dict = {
                    "status": "success",
                    "pet_type_analyzed": "dog",
                    "behavior_analysis": {"detected_behavior": beh, "confidence": b_conf,
                                          "emotion": emo, "emotion_confidence": e_conf},
                    "audio_analysis":    {"detected_sound": "Unknown", "confidence": 0.0},
                    "patella_analysis":  {"status": pat, "confidence": p_conf},
                    "summary": f"[DOG] {beh} (Image Analysis)",
                }
            else:
                beh, b_conf = self._infer_image_task(features, "cat", "behavior", self.cat_behavior_classes)
                emo, e_conf = self._infer_image_task(features, "cat", "emotion",  self.cat_emotion_classes)
                res_dict = {
                    "status": "success",
                    "pet_type_analyzed": "cat",
                    "behavior_analysis": {"detected_behavior": beh, "confidence": b_conf,
                                          "emotion": emo, "emotion_confidence": e_conf},
                    "audio_analysis":    {"detected_sound": "Unknown", "confidence": 0.0},
                    "summary": f"[CAT] {beh} (Image Analysis)",
                }

            return sanitize_data(res_dict)

        except Exception as e:
            print(f"[AI ENGINE] Error in analyze_image: {e}")
            return {"status": "error", "message": f"Image behavior inference failed: {str(e)}"}

    def analyze_hybrid(self, image_data: bytes, audio_data: bytes, pet_type: str = "dog") -> dict:
        """PNG(시각) + MP4 오디오(소리) 분리 분석."""
        self._wait_for_load()
        if not self.is_loaded:
            return {"status": "error", "message": "Behavior models are not loaded yet."}

        pt_label = self._normalize_pet_type(pet_type)
        print(f"[AI ENGINE] Hybrid analysis for pet_type: {pt_label}")

        try:
            img_pil = Image.open(io.BytesIO(image_data)).convert("RGB")
            features = self._siglip_features([img_pil])

            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(audio_data)
                temp_path = tmp.name
            audio_tensor = self.extract_audio_tensor(temp_path)
            os.remove(temp_path)

            if pt_label == "dog":
                beh, b_conf = self._infer_image_task(features, "dog", "behavior", self.dog_behavior_classes)
                emo, e_conf = self._infer_image_task(features, "dog", "emotion",  self.dog_emotion_classes)
                snd, s_conf = self._infer_audio_task(audio_tensor, "dog", self.dog_sound_classes)
                pat, p_conf = self._infer_patella_task(features, self.dog_patella_classes)
                res_dict = {
                    "status": "success",
                    "pet_type_analyzed": "dog",
                    "behavior_analysis": {"detected_behavior": beh, "confidence": b_conf,
                                          "emotion": emo, "emotion_confidence": e_conf},
                    "audio_analysis":    {"detected_sound": snd, "confidence": s_conf},
                    "patella_analysis":  {"status": pat, "confidence": p_conf},
                    "summary": f"[DOG] {beh} with {snd}",
                }
            else:
                beh, b_conf = self._infer_image_task(features, "cat", "behavior", self.cat_behavior_classes)
                emo, e_conf = self._infer_image_task(features, "cat", "emotion",  self.cat_emotion_classes)
                snd, s_conf = self._infer_audio_task(audio_tensor, "cat", self.cat_sound_classes)
                res_dict = {
                    "status": "success",
                    "pet_type_analyzed": "cat",
                    "behavior_analysis": {"detected_behavior": beh, "confidence": b_conf,
                                          "emotion": emo, "emotion_confidence": e_conf},
                    "audio_analysis":    {"detected_sound": snd, "confidence": s_conf},
                    "summary": f"[CAT] {beh} with {snd}",
                }

            return sanitize_data(res_dict)

        except Exception as e:
            print(f"[AI ENGINE] Error in analyze_hybrid: {e}")
            import traceback; traceback.print_exc()
            return {"status": "error", "message": f"Hybrid inference failed: {str(e)}"}

    def analyze_clip(self, video_bytes: bytes, pet_type: str = "dog") -> dict:
        self._wait_for_load()
        if not self.is_loaded:
            return {"status": "error", "message": "Behavior models are not loaded yet."}

        pt_label = self._normalize_pet_type(pet_type)
        print(f"[AI ENGINE] Analyzing clip for pet_type: {pet_type} (Mapped to: {pt_label})")

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(video_bytes)
                temp_path = tmp.name

            frames_pil   = self.extract_frames(temp_path)
            audio_tensor = self.extract_audio_tensor(temp_path)
            os.remove(temp_path)

            features = self._siglip_features(frames_pil) if frames_pil else None

            if pt_label == "dog":
                beh, b_conf = self._infer_image_task(features, "dog", "behavior", self.dog_behavior_classes)
                emo, e_conf = self._infer_image_task(features, "dog", "emotion",  self.dog_emotion_classes)
                snd, s_conf = self._infer_audio_task(audio_tensor, "dog", self.dog_sound_classes)
                pat, p_conf = self._infer_patella_task(features, self.dog_patella_classes)
                res_dict = {
                    "status": "success",
                    "pet_type_analyzed": "dog",
                    "behavior_analysis": {"detected_behavior": beh, "confidence": b_conf,
                                          "emotion": emo, "emotion_confidence": e_conf},
                    "audio_analysis":    {"detected_sound": snd, "confidence": s_conf},
                    "patella_analysis":  {"status": pat, "confidence": p_conf},
                    "summary": f"[DOG] {beh} with {snd}",
                }
            else:
                beh, b_conf = self._infer_image_task(features, "cat", "behavior", self.cat_behavior_classes)
                emo, e_conf = self._infer_image_task(features, "cat", "emotion",  self.cat_emotion_classes)
                snd, s_conf = self._infer_audio_task(audio_tensor, "cat", self.cat_sound_classes)
                res_dict = {
                    "status": "success",
                    "pet_type_analyzed": "cat",
                    "behavior_analysis": {"detected_behavior": beh, "confidence": b_conf,
                                          "emotion": emo, "emotion_confidence": e_conf},
                    "audio_analysis":    {"detected_sound": snd, "confidence": s_conf},
                    "summary": f"[CAT] {beh} with {snd}",
                }

            return sanitize_data(res_dict)

        except Exception as e:
            print(f"[AI ENGINE] Error in analyze_clip: {e}")
            import traceback; traceback.print_exc()
            return {"status": "error", "message": f"Behavior inference failed: {str(e)}"}


daily_behavior_engine = DailyBehaviorEngine()
