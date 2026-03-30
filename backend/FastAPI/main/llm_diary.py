import os
import json
from datetime import datetime, timedelta
from groq import Groq
from firebase_admin import db as firebase_db

# Load .env from backend directory
try:
    from dotenv import load_dotenv
    _env_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
    load_dotenv(os.path.abspath(_env_path))
except ImportError:
    pass  # dotenv not installed; rely on system env vars

# Initialize Groq client
# This assumes GROQ_API_KEY is properly set in the environment (.env)
try:
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
except Exception as e:
    print(f"Failed to initialize Groq client: {e}")
    groq_client = None

def get_daily_logs_for_diary(user_id: str, target_date: str = None) -> list:
    """
    Fetch daily analysis logs for a user on a specific date.
    New structure: users/{user_id}/day/{YYYY-MM-DD}/{HH:MM:SS}/
      └── image_url, analysis_result
    target_date format: YYYY-MM-DD
    """
    if target_date is None:
        target_date = datetime.now().strftime("%Y-%m-%d")

    ref = firebase_db.reference(f'users/{user_id}/day/{target_date}')
    logs_on_day = ref.get() or {}

    logs = []
    for time_key, log in logs_on_day.items():
        if not isinstance(log, dict):
            continue
        # Attach the time key so we can sort; analysis_result must exist
        if "analysis_result" in log:
            log["_time_key"] = time_key  # HH:MM:SS
            logs.append(log)

    # Sort by HH:MM:SS string (lexicographic == chronological)
    logs.sort(key=lambda x: x.get("_time_key", ""))
    return logs

def generate_daily_diary(user_id: str, target_date: str = None) -> str:
    """
    Generates a daily diary using the Groq API based on Firebase RTDB logs.
    """
    if groq_client is None:
        return "Groq API 클라이언트가 초기화되지 않았습니다. API 키를 확인해주세요."
        
    logs = get_daily_logs_for_diary(user_id, target_date)

    if not logs:
        return f"{target_date if target_date else '오늘'}의 기록이 충분하지 않아 일기를 작성하기 어렵습니다."

    # pet_type is no longer in the log; fetch from Firebase pet_info
    try:
        pet_info = firebase_db.reference(f'users/{user_id}/pet_info').get() or {}
        pet_type = pet_info.get("pet_type", "반려동물")
    except Exception:
        pet_type = "반려동물"

    # Sample at most 30 logs to prevent Groq token limit errors (increased from 15 to accommodate 24 points)
    import random
    if len(logs) > 30:
        sampled_logs = random.sample(logs, 30)
        sampled_logs.sort(key=lambda x: x.get("_time_key", ""))
    else:
        sampled_logs = logs
    
    activities_summary = []
    for log in sampled_logs:
        time_str = log.get("_time_key", "??:??:??")[:5]  # HH:MM from HH:MM:SS
        behavior_info = log.get("analysis_result", {})
        
        # Parse nested AI inference structure 
        # based on daily_behavior_inference.py return structure
        if behavior_info.get("status") == "success":
            beh = behavior_info.get("behavior_analysis", {}).get("detected_behavior", "Unknown")
            emo = behavior_info.get("behavior_analysis", {}).get("emotion", "Unknown")
            snd = behavior_info.get("audio_analysis", {}).get("detected_sound", "Unknown")
            
            activity = f"- {time_str}: 주로 {beh} 행동을 보임. 당시 감정은 {emo} 상태로 추정됨."
            if snd and snd != "Unknown" and snd != "None" and snd != "background":
                 activity += f" ({snd} 소리를 냄)"
            
            # Patella log
            patella = behavior_info.get("patella_analysis", {}).get("status")
            if patella and patella != "정상" and patella != "Normal" and patella != "Unknown":
                activity += f" (※ 주의: 슬개골 이상 의심 모션 '{patella}' 포착됨)"
                
            activities_summary.append(activity)

    activities_text = "\n".join(activities_summary)
    
    # ─────────────────────────── Generate Pet Diary (1st Person) ───────────────────────────
    pet_nickname = pet_info.get("pet_name", "나")
    
    pet_diary_prompt = f"""
당신은 이제부터 반려동물 '{pet_nickname}'(종: {pet_type}) 본인입니다. 
아래의 오늘 하루 AI 행동 기록을 바탕으로, 오늘 하루가 어땠는지 당신(반려동물)의 시점에서 일기를 작성하세요.

[오늘의 기록]
{activities_text}

[작성 지침]
1. 반드시 반려동물의 입장에서 1인칭('나', '나의' 등)으로 작성하세요.
2. 주인(보호자)에게 오늘 하루 느낀 감정을 다정하게 이야기하듯 쓰세요.
3. **반드시 3문장 이내로** 매우 간결하게 작성하세요.
4. 말투는 귀엽고 친근하게 하세요.
    """

    # ─────────────────────────── Generate AI Report (Professional 1-liner) ───────────────────────────
    report_prompt = f"""
당신은 전문 반려동물 행동 분석가입니다.
아래의 오늘 하루 AI 행동 기록을 바탕으로, 보호자에게 줄 'AI 행동 분석 레포트'를 작성하세요.

[오늘의 기록]
{activities_text}

[작성 지침]
1. **반드시 딱 한 줄(한 문장)**로만 작성하세요.
2. 어투는 매우 전문적이어야 하며, 반드시 "~필요합니다", "~보입니다" 등으로 정중하고 깔끔하게 끝내세요.
3. 요점만 명확하게 전달하세요.
    """

    try:
        # Step 1: Generate Pet Diary
        diary_resp = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system", 
                    "content": (
                        "당신은 반려동물입니다. 주인에게 말하듯 귀엽게 1인칭 시점('나')으로 일기를 쓰세요. "
                        "반드시 한국어로 120~150자 이내로 작성하세요. "
                        "만약 입력 데이터에 슬개골(무릎) 문제가 있다면, '무릎이 아프다'나 '다리가 무겁다'처럼 "
                        "반려동물이 느낄법한 자연스러운 표현을 사용하여 반드시 언급하세요. "
                        "단, 'AI 탐지', 'Normal', '슬개골 탈구'와 같은 전문적이거나 인공지능스러운 단어는 절대 사용하지 마세요."
                    )
                },
                {"role": "user", "content": pet_diary_prompt}
            ],
            model="llama-3.1-8b-instant",
            temperature=0.8, # 창의적인 표현을 위해 약간 높게 유지
            max_tokens=150,
        )
        pet_diary_content = diary_resp.choices[0].message.content.strip()

        # Step 2: Generate AI Report
        report_resp = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system", 
                    "content": (
                        "당신은 전문 반려동물 행동 분석가입니다. "
                        "반드시 1~2문장 내외로 전문적인 조언을 작성하세요. "
                        "문장은 반드시 '~필요합니다' 또는 '~보입니다'로 끝나야 합니다. "
                        "내용에 '감정'이라는 단어와 '~필요해 보입니다'라는 문구를 반드시 포함하세요. "
                        "만약 슬개골 질환이 의심되는 상황이라면, 반드시 '슬개골'이라는 단어를 포함하여 구체적인 조언을 작성하세요."
                    )
                },
                {"role": "user", "content": report_prompt}
            ],
            model="llama-3.1-8b-instant",
            temperature=0.4, # 일관된 전문가 톤을 위해 온도를 낮춤
            max_tokens=30,
        )
        report_content = report_resp.choices[0].message.content.strip()
        
        # ─────────────────────────── Save to Firebase (New Structure) ───────────────────────────
        # Path: users/{user_id}/Diary/{date}
        diary_ref = firebase_db.reference(f'users/{user_id}/Diary/{target_date}')
        
        # Existing memo preservation (if any)
        existing_data = diary_ref.get() or {}
        memo = existing_data.get("memo", "")
        
        diary_ref.update({
            "pet_diary": pet_diary_content,
            "report": report_content,
            "memo": memo
        })
        
        return pet_diary_content  # Return the pet diary for UI display
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"일기 및 레포트 생성 중 오류가 발생했습니다: {str(e)}"

def get_diary_list(user_id: str, limit: int = 0) -> list:
    """
    Fetches the saved diaries for a user from users/{user_id}/Diary/{date}.
    Each entry includes pet_diary, report, and memo.
    And returns up to 5 random images from the same date in MinIO logs.
    """
    diary_main_ref = firebase_db.reference(f'users/{user_id}/Diary')
    all_diary_entries = diary_main_ref.get() or {}

    # Fetch day logs to find image URLs
    day_ref = firebase_db.reference(f'users/{user_id}/day')
    all_days_logs = day_ref.get() or {}

    diaries = []
    for date_key, diary_data in all_diary_entries.items():
        # Handle new structure (dict) and old structure (string/dict) compatibility
        pet_diary = ""
        report = ""
        memo = ""
        
        if isinstance(diary_data, dict):
            pet_diary = diary_data.get("pet_diary") or diary_data.get("content", "")
            report = diary_data.get("report", "")
            memo = diary_data.get("memo", "")
        elif isinstance(diary_data, str):
            pet_diary = diary_data # legacy string
        
        # Find all available image_urls for this date
        all_image_urls = []
        day_logs = all_days_logs.get(date_key, {})
        if isinstance(day_logs, dict):
            for log in day_logs.values():
                if isinstance(log, dict) and log.get("image_url"):
                    url = log["image_url"].replace("minio:9000", "localhost:9000")
                    all_image_urls.append(url)

        # Pick up to 4 random images
        import random
        selected_images = random.sample(all_image_urls, min(len(all_image_urls), 4)) if all_image_urls else []
        
        # Main thumbnail (first picked or placeholder)
        main_image = selected_images[0] if selected_images else ""

        diaries.append({
            "date": date_key,
            "pet_diary": pet_diary,
            "report": report,
            "memo": memo,
            "image_url": main_image,    # primary image (for list previews)
            "image_urls": selected_images # up to 5 random images
        })

    diaries.sort(key=lambda x: x.get("date", ""), reverse=True)

    if limit > 0:
        return diaries[:limit]
    return diaries
