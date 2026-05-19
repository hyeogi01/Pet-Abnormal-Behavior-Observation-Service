"""
normal_omni_extract.py  (Step 1/2)
────────────────────────────────────────────────────────────────────
SigLIP (이미지) + Whisper (오디오) feature extractor.

- SigLIP-So400m: Qwen2.5-Omni 내부의 동일한 vision encoder (~400M)
  이미지당 10-50ms → 전체 추출 10-30분
- Whisper-small: 강력한 오디오 encoder (244M)
  오디오당 50-100ms
- 결과: features/{species}_{task}_{split}.pt

Requirements:
  pip install torch torchvision torchaudio
  pip install transformers accelerate
  pip install librosa pillow tqdm

Usage:
  python normal_omni_extract.py
────────────────────────────────────────────────────────────────────
"""

import os, gc, json, random, warnings, time
import numpy as np
import torch
import librosa

from PIL import Image, ImageFile
from collections import defaultdict
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# ── Models ────────────────────────────────────────────────────────
SIGLIP_ID  = "google/siglip-so400m-patch14-384"    # vision encoder (1152-dim)
WAV2VEC2_ID = "facebook/wav2vec2-base"                  # audio encoder (768-dim)

# ── Batch sizes (encoder만 사용 → 대량 배치 가능) ────────────────
IMG_BATCH   = 32     # SigLIP: ~400M params, 384x384 → batch 32 OK on 24GB
AUDIO_BATCH = 16     # Whisper: ~244M params

# ── Audio ─────────────────────────────────────────────────────────
SR            = 16000
MAX_AUDIO_LEN = SR * 5  # 5 seconds

# ── Data ──────────────────────────────────────────────────────────
DOG_NORMAL_DIR = "files/work/dog_normal_dataset"
CAT_NORMAL_DIR = "files/work/cat_normal_dataset"
FEATURE_DIR    = "features"
os.makedirs(FEATURE_DIR, exist_ok=True)

# ── Sampling ──────────────────────────────────────────────────────
#    SigLIP은 빠르니까 넉넉하게 사용
MAX_PER_CLASS_TRAIN = {
    "behavior": 2000,
    "emotion":  2000,
    "sound":    None,   # 전부 (이미 적음)
    "patella":  2000,
}
MAX_PER_CLASS_VALTEST = 500

# ── Class definitions ─────────────────────────────────────────────
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

# ═══════════════════════════════════════════════════════════════════
#  PATELLA FEATURES (59-dim, 기존 코드 동일)
# ═══════════════════════════════════════════════════════════════════
PATELLA_KP_LABELS = ["Iliac crest","Femoral greater trochanter","Femorotibial joint",
                     "Lateral malleolus of the distal tibia",
                     "Distal lateral aspect of the fifth metatarsus"]
PATELLA_KP_SLOT = {f"{l}_{i}":idx*2+i for idx,l in enumerate(PATELLA_KP_LABELS) for i in range(2)}
PATELLA_FEAT_DIM = 59

def _angle_deg(a,b,c):
    a,b,c=np.asarray(a,float),np.asarray(b,float),np.asarray(c,float)
    v1,v2=a-b,c-b; n1,n2=np.linalg.norm(v1),np.linalg.norm(v2)
    if n1<1e-8 or n2<1e-8: return 0.0
    return float(np.degrees(np.arccos(np.clip(np.dot(v1,v2)/(n1*n2),-1,1))))

def _parse_patella_features(json_path):
    feat=np.zeros(PATELLA_FEAT_DIM,dtype=np.float32)
    if not os.path.exists(json_path): return feat
    try:
        with open(json_path,encoding="utf-8") as f: data=json.load(f)
        lc,kp=defaultdict(int),{}
        for ann in data.get("annotation_info",[]):
            l=ann["label"];o=lc[l];lc[l]+=1;s=PATELLA_KP_SLOT.get(f"{l}_{o}")
            if s is None:continue
            x,y=float(ann["x"]),float(ann["y"]);b=s*3;feat[b],feat[b+1],feat[b+2]=x,y,1.0;kp[f"{l}_{o}"]=(x,y)
        Z=(0.,0.);angs=[]
        for side in [0,1]:
            il=kp.get(f"Iliac crest_{min(side,lc.get('Iliac crest',0)-1)}",Z)
            tr=kp.get(f"Femoral greater trochanter_{side}",Z);ft=kp.get(f"Femorotibial joint_{side}",Z)
            ma=kp.get(f"Lateral malleolus of the distal tibia_{side}",Z)
            me=kp.get(f"Distal lateral aspect of the fifth metatarsus_{side}",Z)
            angs.append((_angle_deg(il,tr,ft)/180,_angle_deg(tr,ft,ma)/180,_angle_deg(ft,ma,me)/180))
        feat[30],feat[32],feat[34]=angs[0];feat[31],feat[33],feat[35]=angs[1]
        for k,(l,r) in enumerate(zip(angs[0],angs[1])):feat[36+k]=abs(l-r)/(l+r+1e-6)
        for fi,frame in enumerate(data.get("sensor_values",[])[:3]):
            v=np.array(frame,dtype=np.float32);c,r=int(v[4]),int(v[5]);gs=c*r;g=v[7:7+gs]/255.0
            if len(g)<gs:continue
            g2=g.reshape(r,c);L,R=g2[:,:c//2],g2[:,c//2:];Ls,Rs=L.sum(),R.sum();b=39+fi*6
            feat[b:b+6]=[g.mean(),g.max(),g.std(),Ls/(L.size+1e-6),Rs/(R.size+1e-6),abs(Ls-Rs)/(Ls+Rs+1e-6)]
        for rec in data.get("pet_medical_record_info",[]):
            p,v=rec.get("foot_position",""),float(rec.get("value",0))/4.0
            if p=="left":feat[57]=v
            elif p=="right":feat[58]=v
    except:pass
    return feat


# ═══════════════════════════════════════════════════════════════════
#  DATA COLLECTION + SAMPLING
# ═══════════════════════════════════════════════════════════════════
def collect_samples(task_dir, class_list, max_per_class=None, is_patella=False):
    label_map = {c: i for i, c in enumerate(class_list)}
    samples_by_class = defaultdict(list)
    for cls in class_list:
        d = os.path.join(task_dir, cls)
        if not os.path.isdir(d): continue
        for f in os.listdir(d):
            if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                path = os.path.join(d, f)
                if is_patella:
                    jp = os.path.splitext(path)[0] + '.json'
                    samples_by_class[cls].append((path, jp, label_map[cls]))
                else:
                    samples_by_class[cls].append((path, label_map[cls]))

    rng = random.Random(SEED)
    all_samples = []
    for cls in class_list:
        items = samples_by_class[cls]
        if max_per_class and len(items) > max_per_class:
            rng.shuffle(items); items = items[:max_per_class]
            print(f"      {cls}: {len(samples_by_class[cls])} → {len(items)} (sampled)")
        else:
            print(f"      {cls}: {len(items)}")
        all_samples.extend(items)
    return all_samples


def collect_audio_samples(task_dir, class_list, max_per_class=None):
    label_map = {c: i for i, c in enumerate(class_list)}
    samples_by_class = defaultdict(list)
    for cls in class_list:
        d = os.path.join(task_dir, cls)
        if not os.path.isdir(d): continue
        for f in os.listdir(d):
            if f.lower().endswith(('.wav', '.mp3', '.m4a')):
                samples_by_class[cls].append((os.path.join(d, f), label_map[cls]))

    rng = random.Random(SEED)
    all_samples = []
    for cls in class_list:
        items = samples_by_class[cls]
        if max_per_class and len(items) > max_per_class:
            rng.shuffle(items); items = items[:max_per_class]
            print(f"      {cls}: {len(samples_by_class[cls])} → {len(items)} (sampled)")
        else:
            print(f"      {cls}: {len(items)}")
        all_samples.extend(items)
    return all_samples


# ═══════════════════════════════════════════════════════════════════
#  MODEL LOADING
# ═══════════════════════════════════════════════════════════════════
def load_siglip():
    from transformers import AutoModel, SiglipImageProcessor
    print(f"\n📦 Loading SigLIP: {SIGLIP_ID}")
    model = AutoModel.from_pretrained(SIGLIP_ID, torch_dtype=torch.float16)
    model = model.vision_model.to(DEVICE).eval()
    processor = SiglipImageProcessor.from_pretrained(SIGLIP_ID)  # ← 변경
    hidden_dim = model.config.hidden_size
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  ✅ Vision encoder: {params:.0f}M params, hidden_dim={hidden_dim}")
    return model, processor, hidden_dim

def load_wav2vec2():
    """Wav2Vec2 encoder → 768-dim feature per audio."""
    from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
    print(f"\n📦 Loading Wav2Vec2: {WAV2VEC2_ID}")
    model = Wav2Vec2Model.from_pretrained(WAV2VEC2_ID, torch_dtype=torch.float16)
    model = model.to(DEVICE).eval()
    processor = Wav2Vec2FeatureExtractor.from_pretrained(WAV2VEC2_ID)
    hidden_dim = model.config.hidden_size  # 768
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  ✅ Audio encoder: {params:.0f}M params, hidden_dim={hidden_dim}")
    return model, processor, hidden_dim

# ═══════════════════════════════════════════════════════════════════
#  FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════════
def extract_image_features(model, processor, image_paths, batch_size=IMG_BATCH):
    """SigLIP vision encoder: 이미지 → (N, 1152)."""
    all_features = []
    for i in tqdm(range(0, len(image_paths), batch_size), desc="    images"):
        batch_paths = image_paths[i:i+batch_size]
        images = []
        for p in batch_paths:
            try: images.append(Image.open(p).convert("RGB"))
            except: images.append(Image.new("RGB", (384, 384)))

        inputs = processor(images=images, return_tensors="pt", padding=True)
        pixel_values = inputs["pixel_values"].to(DEVICE, dtype=torch.float16)

        with torch.no_grad():
            outputs = model(pixel_values=pixel_values)
            # Mean pool over patch tokens → (B, hidden_dim)
            h = outputs.last_hidden_state.mean(dim=1).float().cpu()

        all_features.append(h)
        del inputs, outputs, h, pixel_values

    return torch.cat(all_features, dim=0)


def extract_audio_features(model, processor, audio_paths, batch_size=AUDIO_BATCH):
    """Wav2Vec2 encoder: 오디오 → (N, 768)."""
    all_features = []
    for i in tqdm(range(0, len(audio_paths), batch_size), desc="    audio"):
        batch_paths = audio_paths[i:i+batch_size]
        waveforms = []
        for p in batch_paths:
            try:
                w, _ = librosa.load(p, sr=SR, mono=True)
                w = w[:MAX_AUDIO_LEN] if len(w) > MAX_AUDIO_LEN \
                    else np.pad(w, (0, MAX_AUDIO_LEN - len(w))).astype(np.float32)
            except:
                w = np.zeros(MAX_AUDIO_LEN, dtype=np.float32)
            waveforms.append(w)

        inputs = processor(waveforms, sampling_rate=SR, return_tensors="pt",
                           padding=True)
        input_values = inputs.input_values.to(DEVICE, dtype=torch.float16)

        with torch.no_grad():
            outputs = model(input_values=input_values)
            # Mean pool over time → (B, hidden_dim)
            h = outputs.last_hidden_state.mean(dim=1).float().cpu()

        all_features.append(h)
        del inputs, outputs, h, input_values

    return torch.cat(all_features, dim=0)


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════
def main():
    siglip_model, siglip_proc, img_dim = load_siglip()
    wav2vec2_model, wav2vec2_proc, aud_dim = load_wav2vec2()

    print(f"\n  Image feature dim: {img_dim}")
    print(f"  Audio feature dim: {aud_dim}")

    tasks = [
        ("dog", DOG_NORMAL_DIR, "behavior", DOG_BEHAVIOR, "image"),
        ("dog", DOG_NORMAL_DIR, "emotion",  DOG_EMOTION,  "image"),
        ("dog", DOG_NORMAL_DIR, "sound",    DOG_SOUND,    "audio"),
        ("dog", DOG_NORMAL_DIR, "patella",  DOG_PATELLA,  "patella"),
        ("cat", CAT_NORMAL_DIR, "behavior", CAT_BEHAVIOR, "image"),
        ("cat", CAT_NORMAL_DIR, "emotion",  CAT_EMOTION,  "image"),
        ("cat", CAT_NORMAL_DIR, "sound",    CAT_SOUND,    "audio"),
    ]

    total_extracted = 0
    t0 = time.time()

    for species, base_dir, task_name, cls_list, modality in tasks:
        tag = f"{species}_{task_name}"

        for split in ["train", "val", "test"]:
            split_dir = os.path.join(base_dir, split, task_name)
            if not os.path.isdir(split_dir):
                print(f"  ⚠️  {tag}/{split} not found, skipping")
                continue

            save_path = os.path.join(FEATURE_DIR, f"{tag}_{split}.pt")
            if os.path.exists(save_path):
                print(f"  ✅ {tag}/{split} already extracted, skipping")
                continue

            spl_max = MAX_PER_CLASS_TRAIN.get(task_name) if split == "train" \
                      else MAX_PER_CLASS_VALTEST

            print(f"\n  📊 {tag}/{split}:")

            if modality == "audio":
                samples = collect_audio_samples(split_dir, cls_list, spl_max)
                paths = [s[0] for s in samples]
                labels = torch.tensor([s[1] for s in samples], dtype=torch.long)
                hidden_dim = aud_dim
                print(f"    Extracting {len(paths)} audio features (Wav2Vec2)...")
                features = extract_audio_features(wav2vec2_model, wav2vec2_proc, paths)
                torch.save({"features": features, "labels": labels,
                            "hidden_dim": hidden_dim, "classes": cls_list,
                            "modality": "audio", "encoder": WAV2VEC2_ID}, save_path)

            elif modality == "patella":
                samples = collect_samples(split_dir, cls_list, spl_max, is_patella=True)
                paths = [s[0] for s in samples]
                patella_feats = torch.stack([
                    torch.from_numpy(_parse_patella_features(s[1])) for s in samples])
                labels = torch.tensor([s[2] for s in samples], dtype=torch.long)
                hidden_dim = img_dim
                print(f"    Extracting {len(paths)} image features (SigLIP)...")
                features = extract_image_features(siglip_model, siglip_proc, paths)
                torch.save({"features": features, "patella_feats": patella_feats,
                            "labels": labels, "hidden_dim": hidden_dim,
                            "classes": cls_list,
                            "modality": "image", "encoder": SIGLIP_ID}, save_path)

            else:  # image
                samples = collect_samples(split_dir, cls_list, spl_max)
                paths = [s[0] for s in samples]
                labels = torch.tensor([s[1] for s in samples], dtype=torch.long)
                hidden_dim = img_dim
                print(f"    Extracting {len(paths)} image features (SigLIP)...")
                features = extract_image_features(siglip_model, siglip_proc, paths)
                torch.save({"features": features, "labels": labels,
                            "hidden_dim": hidden_dim, "classes": cls_list,
                            "modality": "image", "encoder": SIGLIP_ID}, save_path)

            total_extracted += len(features)
            print(f"    💾 {save_path} ({features.shape})")
            gc.collect(); torch.cuda.empty_cache()

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"🎉 Feature extraction complete!")
    print(f"   Total samples: {total_extracted:,}")
    print(f"   Elapsed: {elapsed/60:.1f} min")
    print(f"   Speed: {total_extracted/max(elapsed,1):.1f} samples/sec")
    print(f"   Files: {FEATURE_DIR}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
