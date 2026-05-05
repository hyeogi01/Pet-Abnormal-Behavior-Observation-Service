from fastapi import FastAPI, File, UploadFile, Form
from contextlib import asynccontextmanager
from pydantic import BaseModel
import firebase_admin
from firebase_admin import credentials, db as firebase_db
from fastapi.middleware.cors import CORSMiddleware
import io
import datetime
import uuid
import tempfile

# AI Inference module import
from FastAPI.main.model_inference import ai_engine
# Minio DB Imports
from FastAPI.main.db import get_minio_client, DAILY_BEHAVIOR_BUCKET
from FastAPI.main.daily_behavior_inference import daily_behavior_engine

# LLM Diary & Statistics Imports
from FastAPI.main.llm_diary import generate_daily_diary, get_diary_list
from FastAPI.main.statistics import get_weekly_statistics

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load AI models in background to avoid blocking login/signup
    import threading
    
    def load_models_concurrently():
        print("[INIT] Startup: Loading heavy AI models in background...")
        try:
            ai_engine.load_models()
            daily_behavior_engine.load_models()
            print("[INIT] Startup: All AI models loaded successfully and ready.")
        except Exception as e:
            print(f"[INIT] Startup Error: Failed to load models in background: {e}")

    # Start loading in a separate thread
    load_thread = threading.Thread(target=load_models_concurrently)
    load_thread.start()
    
    # Initialize DB (fast)
    get_minio_client()
    
    yield
    print("Shutting down...")

app = FastAPI(lifespan=lifespan)

# 1. 파이어베이스 초기화
import os as _os
_KEY_PATH = _os.path.join(_os.path.dirname(__file__), '..', 'key', 'testApi.json')
cred = credentials.Certificate(_os.path.abspath(_KEY_PATH))
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://test-25cac-default-rtdb.asia-southeast1.firebasedatabase.app/'
})

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # 모든 도메인 허용
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/ready/")
def check_ready():
    # Return loading status of the daily behavior engine (it includes omni models)
    return {
        "status": "ready" if daily_behavior_engine.is_loaded else "loading",
        "detail": "AI models are still initializing in the background." if not daily_behavior_engine.is_loaded else "All systems ready."
    }

# 사용자 로그인
class User(BaseModel):
    user_id: str
    password: str

@app.post("/login/")
def login(user: User):
    try:
        ref = firebase_db.reference(f'users/{user.user_id}')
        user_data = ref.get()

        if not user_data:
            return {"status": "error", "message": "아이디가 존재하지 않습니다."}

        # 비밀번호 체크 (딕셔너리 또는 문자열 구조 모두 대응)
        fb_password = ""
        if isinstance(user_data, dict):
            fb_password = user_data.get('password')
        else:
            fb_password = user_data # 단순 문자열인 경우

        if fb_password == user.password:
            # Check if pet_info exists in Firebase
            has_pet_info = False
            if isinstance(user_data, dict):
                has_pet_info = 'pet_info' in user_data

            return {
                "status": "success",
                "message": "로그인 성공",
                "user_id": user.user_id,
                "has_pet_info": has_pet_info
            }
        else:
            return {"status": "error", "message": "아이디 또는 비밀번호가 틀렸습니다."}
            
    except Exception as e:
        print(f"Login error: {str(e)}")
        return {"status": "error", "message": f"서버 오류가 발생했습니다: {str(e)}"}

@app.post("/signup/")
async def signup(user: User):
    # 회원가입 (중복 체크 생략)
    ref = firebase_db.reference(f'users/{user.user_id}')
    # 이미 가입된 아이디인지 확인
    if ref.get() is not None:
        return {"status": "error", "message": "이미 존재하는 아이디입니다."}

    ref.set({
        "password": user.password
    })
    
    return {"status": "success"}

# 테스트 용
class Log(BaseModel):
    pet_name: str
    behavior: str # 예: "짖음", "이상 보행" 등
    timestamp: str

@app.post("/log/")
def save_log(log: Log):
    # 'pet_logs'라는 경로에 데이터 저장
    ref = firebase_db.reference('pet_logs')
    new_log_ref = ref.push(log.dict())
    return {"status": "success", "id": new_log_ref.key}
    
@app.get("/logs/")
def get_logs():
    ref = firebase_db.reference('pet_logs')
    return ref.get()

# 사용자 입력 : 반려동물 기본 정보 PetRegistrationPage
class PetInfo(BaseModel):
    pet_name: str
    pet_type: str
    pet_gender: str
    pet_birthday : str
    
@app.post("/user-input/{user_id}")
def save_pet_info(user_id: str, data: PetInfo):
    # 파이어베이스에 pet_info 저장
    ref = firebase_db.reference(f'users/{user_id}')
    ref.update({"pet_info": data.model_dump()})
    return {"status": "success", "user_id": user_id}

@app.get("/user-pet-info/{user_id}")
def get_all_pet_info(user_id: str):
    ref = firebase_db.reference(f'users/{user_id}/pet_info')
    pet_info = ref.get()

    if pet_info:
        return {
            "status": "success",
            "data": pet_info  # pet_name, pet_type, pet_gender, pet_birthday
        }
    else:
        return {"status": "error", "message": "반려동물 정보가 없습니다."}

# ─────────────────────────── 직접 로그 저장 (Flutter → Firebase) ───────────────────────────
class DirectLogRequest(BaseModel):
    user_id: str
    pet_type: str
    timestamp: str  # ISO 8601
    analysis_result: dict  # AI 분석 결과 JSON (behavior, audio, patella)
    video_url: str = ""  # optional

@app.post("/api/save-log")
def save_log_direct(req: DirectLogRequest):
    """
    Saves a pre-computed analysis result to Firebase RTDB under
    users/{user_id}/day/{YYYY-MM-DD}/{push_key}/.
    Used by the Flutter app to save test or real analysis data without uploading video.
    """
    try:
        log_time = datetime.datetime.fromisoformat(req.timestamp)
        date_str = log_time.strftime("%Y-%m-%d")
        time_str = log_time.strftime("%H:%M:%S")  # 분석 시각 (HH:MM:SS)

        ref = firebase_db.reference(f'users/{req.user_id}/day/{date_str}/{time_str}')
        ref.set({
            "image_url": req.video_url,
            "analysis_result": req.analysis_result
        })
        return {"status": "success", "time": time_str, "date": date_str, "user_id": req.user_id}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# AI 질환 분석 API
@app.post("/api/analyze-disease")
async def analyze_disease(
    user_id: str = Form(...),
    pet_type: str = Form(...),
    disease_type: str = Form(...),
    file: UploadFile = File(...)
):
    import uuid
    import datetime
    import io
    try:
        contents = await file.read()
        
        # ai_engine 추론 로직 호출
        result = ai_engine.analyze(
            image_bytes=contents,
            pet_type=pet_type,
            disease_type=disease_type
        )
        
        # Upload object to MinIO
        minio_client = get_minio_client()
        file_id = str(uuid.uuid4())
        object_name = f"{user_id}/examination/{disease_type}/{file_id}.jpg"
        
        minio_client.put_object(
            DAILY_BEHAVIOR_BUCKET,
            object_name,
            io.BytesIO(contents),
            length=len(contents),
            content_type="image/jpeg"
        )
        image_url = f"http://localhost:9000/{DAILY_BEHAVIOR_BUCKET}/{object_name}"
        
        if isinstance(result, dict) and result.get("status") == "success":
            result["image_url"] = image_url
            
            # Save to Firebase
            log_time = datetime.datetime.now()
            date_str = log_time.strftime("%Y-%m-%d")
            time_str = log_time.strftime("%H:%M:%S")
            
            ref = firebase_db.reference(f'users/{user_id}/Examination_Results/{file_id}')
            ref.set({
                "date": date_str,
                "time": time_str,
                "timestamp": log_time.isoformat(),
                "category": disease_type,
                "result": result,
                "image_url": image_url
            })
            
        return result
    except Exception as e:
        return {"status": "error", "message": f"Server processing error: {str(e)}"}

@app.get("/api/examination-history/{user_id}")
async def get_examination_history(user_id: str):
    try:
        ref = firebase_db.reference(f'users/{user_id}/Examination_Results')
        results = ref.get()
        if not results:
            return {"status": "success", "data": []}
            
        history = []
        for file_id, data in results.items():
            if isinstance(data, dict):
                data["id"] = file_id
                history.append(data)
            
        # Sort by timestamp descending
        history.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return {"status": "success", "data": history}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# ─────────────────────────── HYBRID NEW ───────────────────────────
# AI 일상 행동 분석 API (Video Clip -> Frame Image Upload)
@app.post("/api/daily-behavior")
def analyze_daily_behavior(
    user_id: str = Form(...),
    pet_type: str = Form(...),
    file: UploadFile = File(...),
    timestamp: str = Form(None)
):
    import tempfile
    import os
    import cv2
    import traceback
    try:
        contents = file.file.read()
        filename = file.filename.lower()
        is_image = filename.endswith(('.jpg', '.jpeg', '.png'))
        
        print(f"Received file: {filename}, len={len(contents)} timestamp={timestamp}", flush=True)
        
        if is_image:
            # 1. Image Inference
            print("Starting image AI inference...", flush=True)
            ai_result = daily_behavior_engine.analyze_image(contents, pet_type)
            image_bytes = contents
            print("Completed image AI inference.", flush=True)
        else:
            # 1. Video Clip Inference
            print("Starting video AI inference...", flush=True)
            ai_result = daily_behavior_engine.analyze_clip(contents, pet_type)
            print("Completed video AI inference.", flush=True)

            # 2. Extract ONLY ONE frame for video
            print("Extracting frame from video...", flush=True)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4", mode="wb") as tmp_file:
                tmp_file.write(contents)
                temp_video_path = tmp_file.name

            cap = cv2.VideoCapture(temp_video_path)
            ret, frame = cap.read()
            cap.release()
            os.remove(temp_video_path)
            
            if ret:
                _, buffer = cv2.imencode('.jpg', frame)
                image_bytes = buffer.tobytes()
            else:
                image_bytes = b""

        # 3. Upload image to MinIO
        minio_client = get_minio_client()
        object_name = f"{user_id}/{uuid.uuid4()}.jpg"
        
        minio_client.put_object(
            DAILY_BEHAVIOR_BUCKET,
            object_name,
            io.BytesIO(image_bytes),
            length=len(image_bytes),
            content_type="image/jpeg"
        )
        
        image_url = f"http://localhost:9000/{DAILY_BEHAVIOR_BUCKET}/{object_name}"
        
        # Determine timestamp
        if timestamp:
            try:
                log_time = datetime.datetime.fromisoformat(timestamp)
            except ValueError:
                log_time = datetime.datetime.now()
        else:
            log_time = datetime.datetime.now()

        # 4. Save to Firebase RTDB
        date_str = log_time.strftime("%Y-%m-%d")
        time_str = log_time.strftime("%H:%M:%S")
        ref = firebase_db.reference(f'users/{user_id}/day/{date_str}/{time_str}')
        ref.set({
            "image_url": image_url,
            "analysis_result": ai_result
        })
        
        return {
            "status": "success",
            "message": "Daily analysis saved successfully",
            "video_url": image_url,
            "ai_inference": ai_result
        }
        
    except Exception as e:
        return {"status": "error", "message": f"Daily behavior processing error: {str(e)}"}

@app.post("/api/simulate-full-day")
async def simulate_full_day(
    user_id: str = Form(...),
    pet_type: str = Form(...)
):
    """
    SIMULATION ONLY: Extracts 24 clips (4s each) from sample1.mp4, analyzes them,
    saves to DB, and returns a generated LLM diary.
    """
    import os
    import uuid
    import datetime
    import io
    from moviepy import VideoFileClip
    
    SIM_SAMPLE_FILE = "sample1.mp4" 
    sim_possible_paths = [
        SIM_SAMPLE_FILE,
        os.path.join(os.path.dirname(__file__), SIM_SAMPLE_FILE),
        os.path.join(os.path.dirname(__file__), "..", "..", SIM_SAMPLE_FILE),
        "/app/sample1.mp4"
    ]
    
    sim_found_path = None
    for p in sim_possible_paths:
        if os.path.exists(p):
            sim_found_path = p
            break
            
    if not sim_found_path:
         # Local check as fallback
        if os.path.exists("backend/sample1.mp4"): sim_found_path = "backend/sample1.mp4"
        else: return {"status": "error", "message": f"Sample video not found. Tried: {sim_possible_paths}"}
    
    target_video = sim_found_path
    normalized_pt = pet_type.lower().strip()

    try:
        print(f"Starting simulation for user {user_id} using {target_video} (PT={normalized_pt})")
        full_clip = VideoFileClip(target_video)
        duration = full_clip.duration
        
        NUM_SIM_CHUNKS = 24
        sim_stride = max(1, duration // NUM_SIM_CHUNKS)
        
        sim_today = datetime.datetime.now().strftime("%Y-%m-%d")
        minio_client = get_minio_client()

        for i in range(NUM_SIM_CHUNKS):
            center_t = i * sim_stride
            start_t = max(0, center_t - 2)
            end_t = min(duration, center_t + 2)
            
            print(f"[{i+1}/24] Extracting clip {start_t:.1f}s - {end_t:.1f}s...")
            
            # Temporary file for the 4s clip
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_clip_file:
                tmp_clip_path = tmp_clip_file.name
            
            sub_clip = full_clip.subclipped(start_t, end_t)
            sub_clip.write_videofile(tmp_clip_path, codec="libx264", audio_codec="aac", logger=None)
            
            with open(tmp_clip_path, "rb") as f:
                clip_bytes = f.read()
            
            # 1. AI Analysis (Clip Analysis)
            print(f"Analyzing clip {i+1}/24 (pet_type={normalized_pt})...")
            sim_ai_result = daily_behavior_engine.analyze_clip(clip_bytes, normalized_pt)
            
            # 2. Extract ONE frame for thumbnail
            import cv2
            cap = cv2.VideoCapture(tmp_clip_path)
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                _, buffer = cv2.imencode('.jpg', frame)
                thumb_image_bytes = buffer.tobytes()
            else:
                thumb_image_bytes = b""

            # 3. MinIO Upload (Thumbnail image)
            object_name = f"{user_id}/sim_{uuid.uuid4()}.jpg"
            minio_client.put_object(
                DAILY_BEHAVIOR_BUCKET,
                object_name,
                io.BytesIO(thumb_image_bytes),
                length=len(thumb_image_bytes),
                content_type="image/jpeg"
            )
            image_url = f"http://localhost:9000/{DAILY_BEHAVIOR_BUCKET}/{object_name}"
            
            # 3-1. Extract Audio to MP3
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_audio_file:
                tmp_audio_path = tmp_audio_file.name
            
            try:
                if sub_clip.audio is not None:
                    sub_clip.audio.write_audiofile(tmp_audio_path, logger=None)
                    with open(tmp_audio_path, "rb") as f:
                        audio_bytes = f.read()
                    
                    # 3-2. Upload MP3 to MinIO
                    audio_object_name = f"{user_id}/sim_{uuid.uuid4()}.mp3"
                    minio_client.put_object(
                        DAILY_BEHAVIOR_BUCKET,
                        audio_object_name,
                        io.BytesIO(audio_bytes),
                        length=len(audio_bytes),
                        content_type="audio/mpeg"
                    )
                    audio_url = f"http://localhost:9000/{DAILY_BEHAVIOR_BUCKET}/{audio_object_name}"
                else:
                    audio_url = ""
            except Exception as ae:
                print(f"Audio extraction/upload error: {ae}")
                audio_url = ""
            finally:
                if os.path.exists(tmp_audio_path):
                    os.remove(tmp_audio_path)

            # 4. Firebase Save
            chunk_time = datetime.datetime.now().replace(hour=i % 24, minute=0, second=0, microsecond=0)
            time_str = chunk_time.strftime("%H:%M:%S")
            full_timestamp = chunk_time.isoformat()
            
            ref = firebase_db.reference(f'users/{user_id}/day/{sim_today}/{time_str}')
            ref.set({
                "video_url": audio_url, # Now pointing to MP3 for verification
                "image_url": image_url, # Still pointing to thumbnail
                "timestamp": full_timestamp,
                "analysis_result": sim_ai_result
            })
            
            # Cleanup temp clip
            os.remove(tmp_clip_path)
            
        full_clip.close()
        print(f"Simulation loop successfully finished for {user_id}")
        
        # 5. Generate Diary
        diary_content = generate_daily_diary(user_id, sim_today)
        
        return {
            "status": "success",
            "message": "Full day simulation (with Audio) completed and diary generated",
            "diary": diary_content
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": f"Simulation failed: {str(e)}"}

# ─────────────────────────── PHASE 3: AI DIARY & STATISTICS ───────────────────────────

@app.get("/api/daily-diary/{user_id}")
async def get_daily_diary(user_id: str, date: str = None):
    """
    Returns a daily diary generated by Groq (LLaMA3) based on the user's pet's daily logs.
    date format: YYYY-MM-DD
    """
    try:
        diary_content = generate_daily_diary(user_id, date)
        if any(keyword in diary_content for keyword in ["오류", "어렵습니다", "초기화되지 않았습니다", "실패"]):
            return {"status": "error", "message": diary_content}
        
        return {
            "status": "success",
            "user_id": user_id,
            "date": date if date else datetime.datetime.now().strftime("%Y-%m-%d"),
            "diary": diary_content
        }
    except Exception as e:
        return {"status": "error", "message": f"Diary generation error: {str(e)}"}

@app.get("/api/statistics/{user_id}")
async def get_pet_statistics(user_id: str, pet_type: str):
    """
    Returns weekly aggregated statistics for the pet based on daily logs.
    Includes emotion charts and patella warnings (if dog).
    """
    try:
        stats = get_weekly_statistics(user_id, pet_type)
        return {
            "status": "success",
            "user_id": user_id,
            "pet_type": pet_type,
            "statistics": stats
        }
    except Exception as e:
        return {"status": "error", "message": f"Stats error: {str(e)}"}

@app.get("/api/daily-stats/{user_id}")
def get_daily_emotion_stats(user_id: str, date: str = None):
    """
    특정 날짜의 감정 데이터를 집계하여 비율을 반환합니다.
    """
    if date is None:
        date = datetime.datetime.now().strftime("%Y-%m-%d")
        
    try:
        ref = firebase_db.reference(f'users/{user_id}/day/{date}')
        logs = ref.get() or {}
        
        if not logs:
            return {"status": "success", "data": {}}
            
        emotion_counts = {}
        total_valid = 0
        
        # 감정 영문 -> 한글 매핑 (프론트엔드와 일치)
        MOOD_MAPPING = {
            "happy": "행복",
            "relaxed": "행복",
            "active": "활발",
            "anxious": "불안",
            "sad": "우울",
            "angry": "화남",
            "bored": "심심",
            "sleepy": "졸림",
            "Unknown": "기타"
        }
        
        for time_key, log in logs.items():
            if not isinstance(log, dict): continue
            
            # analysis_result 또는 behavior_analysis 직접 참조 시도
            res = log.get("analysis_result", {})
            behavior = res.get("behavior_analysis", {}) if isinstance(res, dict) else {}
            
            # 구조가 다른 경우 대비
            raw_emo = behavior.get("emotion") or res.get("emotion") or "Unknown"
            
            # pet_type 접두어 제거 (dog_happy -> happy)
            clean_emo = str(raw_emo).split('_')[-1] if '_' in str(raw_emo) else str(raw_emo)
            
            translated = MOOD_MAPPING.get(clean_emo.lower(), clean_emo)
            emotion_counts[translated] = emotion_counts.get(translated, 0) + 1
            total_valid += 1
                
        # 비율 계산
        emotion_stats = {}
        if total_valid > 0:
            for emo, count in emotion_counts.items():
                emotion_stats[emo] = round((count / total_valid) * 100, 1)
                
        return {
            "status": "success",
            "date": date,
            "total_count": total_valid,
            "data": emotion_stats
        }
    except Exception as e:
        return {"status": "error", "message": f"Stats error: {str(e)}"}

@app.post("/api/analyze-patella/{user_id}")
async def analyze_patella(
    user_id: str,
    file: UploadFile = File(...),
    pet_type: str = Form(default="dog")
):
    """
    사용자가 업로드한 이미지 또는 동영상을 받아 슬개골 AI 분석을 수행합니다.
    - file: 이미지(.jpg, .png) 또는 동영상(.mp4, .mov) 파일
    - pet_type: 'dog' 또는 'cat' (기본값: dog)
    """
    try:
        file_bytes = await file.read()
        filename = file.filename or ""
        content_type = file.content_type or ""
        
        is_video = (
            content_type.startswith("video/") or
            any(filename.lower().endswith(ext) for ext in [".mp4", ".mov", ".avi", ".mkv"])
        )
        
        if is_video:
            result = daily_behavior_engine.analyze_clip(file_bytes, pet_type=pet_type)
        else:
            result = daily_behavior_engine.analyze_image(file_bytes, pet_type=pet_type)
        
        if result.get("status") == "error":
            return {"status": "error", "message": result.get("message", "분석 실패")}
        
        patella = result.get("patella_analysis", {})
        patella_status = patella.get("status", "unknown")
        patella_confidence = patella.get("confidence", 0.0)
        
        # 등급 텍스트 변환
        if patella_status == "normal":
            grade_text = "정상"
            severity = 0
        else:
            try:
                severity = int(patella_status)
                grade_text = f"슬개골 질환 {severity}기 의심"
            except (ValueError, TypeError):
                grade_text = f"이상 감지: {patella_status}"
                severity = 1
        
        return {
            "status": "success",
            "file_type": "video" if is_video else "image",
            "pet_type": pet_type,
            "patella_status": patella_status,
            "patella_confidence": patella_confidence,
            "severity": severity,
            "grade_text": grade_text,
            "behavior_analysis": result.get("behavior_analysis", {}),
        }
        
    except Exception as e:
        print(f"[PATELLA API] Error: {e}")
        return {"status": "error", "message": f"분석 중 오류 발생: {str(e)}"}

@app.get("/api/day-logs/{user_id}")
def get_day_logs(user_id: str, date: str = None):
    """
    Returns the raw logs (abnormal behaviors, etc) for a specific date.
    If the requested date has no data, it falls back to the most recent date that has data.
    """
    requested_date = date if date else datetime.datetime.now().strftime("%Y-%m-%d")
        
    try:
        day_ref = firebase_db.reference(f'users/{user_id}/day')
        all_days = day_ref.get()
        
        if not all_days:
            return {"status": "success", "date": requested_date, "data": {}}
            
        # 요청한 날짜에 데이터가 있는지 확인
        if requested_date in all_days and all_days[requested_date]:
            return {
                "status": "success",
                "date": requested_date,
                "data": all_days[requested_date]
            }
            
        # 요청한 날짜에 데이터가 없으면, 빈 데이터를 반환 (Fallback 제거)
        print(f"[DEBUG API] /api/day-logs/ user_id: {user_id}, requested: {requested_date}, No data found (Fallback removed)", flush=True)
        return {
            "status": "success",
            "date": requested_date,
            "data": {}
        }
    except Exception as e:
        return {"status": "error", "message": f"Fetching day logs error: {str(e)}"}

@app.get("/api/daily-diaries/{user_id}")
def fetch_diary_list(user_id: str, limit: int = 0):
    """
    Fetches a list of generated diaries for the dashboard (limit=5) or total view (limit=0).
    Now returns pet_diary, report, memo, and multiple image_urls.
    """
    try:
        diaries = get_diary_list(user_id, limit)
        return {
            "status": "success",
            "user_id": user_id,
            "data": diaries
        }
    except Exception as e:
        return {"status": "error", "message": f"Fetching diaries error: {str(e)}"}

class MemoRequest(BaseModel):
    user_id: str
    date: str
    memo: str

@app.post("/api/save-memo")
def save_memo(req: MemoRequest):
    """
    Saves or updates the protector's memo for a specific date in the Diary node.
    """
    try:
        ref = firebase_db.reference(f'users/{req.user_id}/Diary/{req.date}')
        ref.update({"memo": req.memo})
        return {"status": "success", "message": "Memo saved successfully"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/api/gallery/{user_id}")
def get_video_gallery(user_id: str):
    """
    Fetches video URLs from users/{user_id}/day/* to display in the Photo Gallery.
    """
    try:
        day_ref = firebase_db.reference(f'users/{user_id}/day')
        all_days = day_ref.get() or {}
        
        gallery_items = []
        for date_key, logs_on_day in all_days.items():
            if not isinstance(logs_on_day, dict):
                continue
            for push_key, doc in logs_on_day.items():
                if isinstance(doc, dict):
                    # Safely get the analysis results
                    beh_info = doc.get("analysis_result", {})
                    if isinstance(beh_info, dict) and beh_info.get("status") == "success":
                        beh_data = beh_info.get("behavior_analysis", {})
                        emotion = beh_data.get("emotion", "Unknown") if isinstance(beh_data, dict) else "Unknown"
                    else:
                        emotion = "Unknown"
                        
                    timestamp_str = doc.get("timestamp", "")
                    
                    # Format timestamp for display
                    display_time = "Unknown"
                    if timestamp_str:
                        try:
                            dt = datetime.datetime.fromisoformat(timestamp_str)
                            display_time = dt.strftime("%Y-%m-%d %H:%M:%S")
                        except ValueError:
                            display_time = timestamp_str

                    url = doc.get("video_url") or doc.get("image_url", "")
                    url = url.replace("minio:9000", "localhost:9000")
                    
                    gallery_items.append({
                        "timestamp": display_time,
                        "video_url": url,
                        "emotion": emotion,
                        "is_image": "video_url" not in doc,
                        "_raw_time": timestamp_str
                    })
                
        # Sort by timestamp descending
        gallery_items.sort(key=lambda x: x.get("_raw_time", ""), reverse=True)
        
        # Remove sorting key
        for item in gallery_items:
            item.pop("_raw_time", None)
            
        return {
            "status": "success",
            "user_id": user_id,
            "data": gallery_items
        }
    except Exception as e:
        return {"status": "error", "message": f"Gallery fetching error: {str(e)}"}

# ─────────────────────────── MYPAGE APIs ───────────────────────────

@app.post("/api/profile-image/{user_id}")
async def upload_profile_image(user_id: str, is_cover: str = Form("false"), file: UploadFile = File(...)):
    import uuid
    import io
    try:
        ref = firebase_db.reference(f'users/{user_id}/pet_info')
        user_pet_info = ref.get() or {}
        minio_client = get_minio_client()
        
        # 1. 기존 이미지가 있다면 MinIO에서 삭제
        old_image_url = user_pet_info.get("cover_image_url") if is_cover.lower() == 'true' else user_pet_info.get("profile_image_url")
        if old_image_url and "user-profiles/" in old_image_url:
            old_object_name = old_image_url.split("user-profiles/")[-1]
            try:
                minio_client.remove_object("user-profiles", old_object_name)
            except Exception as e:
                print(f"Old image deletion failed: {e}")

        # 2. 새 이미지 업로드
        contents = await file.read()
        file_ext = file.filename.split('.')[-1] if '.' in file.filename else 'jpg'
        object_name = f"{user_id}/{uuid.uuid4()}.{file_ext}"
        
        minio_client.put_object(
            "user-profiles",
            object_name,
            io.BytesIO(contents),
            length=len(contents),
            content_type=file.content_type or "image/jpeg"
        )
        image_url = f"http://localhost:9000/user-profiles/{object_name}"
        
        ref = firebase_db.reference(f'users/{user_id}/pet_info')
        if is_cover.lower() == 'true':
            ref.update({"cover_image_url": image_url})
        else:
            ref.update({"profile_image_url": image_url})
            
        return {"status": "success", "image_url": image_url}
    except Exception as e:
        return {"status": "error", "message": str(e)}

class PetNameUpdate(BaseModel):
    pet_name: str

@app.post("/api/update-pet-info/{user_id}")
def update_pet_info(user_id: str, data: PetNameUpdate):
    try:
        ref = firebase_db.reference(f'users/{user_id}/pet_info')
        ref.update({"pet_name": data.pet_name})
        return {"status": "success"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

class WeightRecord(BaseModel):
    date: str
    weight: float

@app.post("/api/weight/{user_id}")
def add_weight(user_id: str, data: WeightRecord):
    try:
        import datetime
        timestamp_key = str(int(datetime.datetime.now().timestamp() * 1000))
        ref = firebase_db.reference(f'users/{user_id}/weight_history/{timestamp_key}')
        ref.set(data.model_dump())
        return {"status": "success"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/api/weight/{user_id}")
def get_weight(user_id: str):
    try:
        ref = firebase_db.reference(f'users/{user_id}/weight_history')
        weights = ref.get() or {}
        
        history = []
        for key, val in weights.items():
            if isinstance(val, dict):
                history.append(val)
                
        # Sort by date
        history.sort(key=lambda x: x.get("date", ""), reverse=False)
        return {"status": "success", "data": history}
    except Exception as e:
        return {"status": "error", "message": str(e)}

class PasswordChangeRequest(BaseModel):
    user_id: str
    current_password: str
    new_password: str

@app.post("/api/change-password/")
def change_password(data: PasswordChangeRequest):
    try:
        ref = firebase_db.reference(f'users/{data.user_id}')
        user_data = ref.get()
        if not user_data:
            return {"status": "error", "message": "유저 정보를 찾을 수 없습니다."}
            
        fb_pw = user_data.get('password') if isinstance(user_data, dict) else user_data
        
        if fb_pw != data.current_password:
            return {"status": "error", "message": "현재 비밀번호가 일치하지 않습니다."}
            
        if isinstance(user_data, dict):
            ref.update({"password": data.new_password})
        else:
            ref.set(data.new_password)
            
        return {"status": "success"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

class DeleteAccountRequest(BaseModel):
    user_id: str

@app.post("/api/delete-account/")
def delete_account(data: DeleteAccountRequest):
    try:
        ref = firebase_db.reference(f'users/{data.user_id}')
        ref.delete()
        return {"status": "success"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# ─────────────────────────── USER SETTINGS ───────────────────────────

class UserSettings(BaseModel):
    recording_interval: int = 60  # 20~60분, 10분 단위
    diary_cover_type: str = "happy"  # "happy" 또는 "frequent"

@app.post("/api/settings/{user_id}")
def save_user_settings(user_id: str, data: UserSettings):
    """
    사용자 설정을 Firebase에 저장합니다.
    - recording_interval: 촬영 주기 (20~60분, 10분 단위)
    - diary_cover_type: 일기 대표 사진 기준 ("happy" 또는 "frequent")
    """
    try:
        # Validate recording_interval
        if data.recording_interval not in [20, 30, 40, 50, 60]:
            return {"status": "error", "message": "촬영 주기는 20, 30, 40, 50, 60분만 설정 가능합니다."}
        if data.diary_cover_type not in ["happy", "frequent"]:
            return {"status": "error", "message": "대표 사진 설정은 'happy' 또는 'frequent'만 가능합니다."}

        ref = firebase_db.reference(f'users/{user_id}/settings')
        ref.set(data.model_dump())
        return {"status": "success", "settings": data.model_dump()}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/api/settings/{user_id}")
def get_user_settings(user_id: str):
    """
    사용자 설정을 Firebase에서 조회합니다.
    설정이 없을 경우 기본값(recording_interval=60, diary_cover_type='happy')을 반환합니다.
    """
    try:
        ref = firebase_db.reference(f'users/{user_id}/settings')
        settings = ref.get()
        if not settings:
            default_settings = {"recording_interval": 60, "diary_cover_type": "happy"}
            return {"status": "success", "settings": default_settings}
        return {"status": "success", "settings": settings}
    except Exception as e:
        return {"status": "error", "message": str(e)}
