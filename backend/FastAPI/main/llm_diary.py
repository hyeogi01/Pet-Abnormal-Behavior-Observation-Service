import os
import json
from datetime import datetime, timedelta
from groq import Groq
from FastAPI.main.db import get_db

# Initialize Groq client
# This assumes GROQ_API_KEY is properly set in the environment (.env)
try:
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
except Exception as e:
    print(f"Failed to initialize Groq client: {e}")
    groq_client = None

def get_daily_logs_for_diary(user_id: str, target_date: str = None) -> list:
    """
    Fetch daily video logs for a user on a specific date.
    target_date format: YYYY-MM-DD
    """
    db_client = get_db()
    
    if target_date is None:
        target_date = datetime.now().strftime("%Y-%m-%d")
        
    start_time = datetime.strptime(target_date, "%Y-%m-%d")
    end_time = start_time + timedelta(days=1)
    
    # Query MongoDB for logs belonging to the user within the target date
    query = {
        "user_id": user_id,
        "timestamp": {
            "$gte": start_time,
            "$lt": end_time
        }
    }
    
    logs = list(db_client["daily_logs"].find(query).sort("timestamp", 1))
    return logs

def generate_daily_diary(user_id: str, target_date: str = None) -> str:
    """
    Generates a daily diary using the Groq API based on MongoDB logs.
    """
    if groq_client is None:
        return "Groq API 클라이언트가 초기화되지 않았습니다. API 키를 확인해주세요."
        
    logs = get_daily_logs_for_diary(user_id, target_date)
    
    if not logs:
        return f"{target_date if target_date else '오늘'}의 기록이 충분하지 않아 일기를 작성하기 어렵습니다."
    
    # Extract prompt data from logs
    pet_type = logs[0].get("pet_type", "반려동물")
    
    # ---------------------------------------------------------
    # RATE LIMIT FIX: If there are too many logs (e.g., 100 chunks),
    # sample at most 15 logs to prevent Groq Token limit errors.
    import random
    if len(logs) > 15:
        sampled_logs = random.sample(logs, 15)
        sampled_logs.sort(key=lambda x: x.get("timestamp", datetime.now()))
    else:
        sampled_logs = logs
    # ---------------------------------------------------------
    
    activities_summary = []
    for log in sampled_logs:
        time_str = log["timestamp"].strftime("%H:%M")
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
    
    prompt = f"""
다음은 {pet_type}의 오늘 하루 동안의 AI 행동 관찰 기록입니다.
이 기록들을 바탕으로 반려동물의 하루를 묘사하는 일기를 하나 작성해주세요.

[오늘의 기록]
{activities_text}

[작성 지침]
1. 당신은 보호자에게 반려동물의 행동과 상태를 다정하고 전문적으로 알려주는 행동 분석 선생님입니다.
2. 어투는 반드시 선생님 말투 (예: "~했습니다", "~로 보입니다", "~군요", "~를 권장합니다" 등)를 확고하게 사용하세요.
3. 보호자를 안심시키면서도, 주의가 필요한 부분(예: 슬개골 이상 의심 등)이 있다면 잘 짚어주세요.
4. 너무 딱딱하지 않고, 동물의 사랑스러움이 묻어나는 따뜻한 톤을 유지하세요.
5. 마크다운 형식을 최소화하고, 자연스러운 줄글 형태로 작성하세요.
    """

    try:
        response = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "당신은 반려동물의 행동을 분석하고 보호자에게 알려주는 친절하고 전문적인 선생님입니다. 항상 한국어로 답변하며, 선생님 특유의 정중하고 따뜻한 말투(~했습니다, ~군요, ~를 권장합니다)를 사용해야 합니다."
                },
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="llama-3.1-8b-instant",  # Updated to the newer 8B model on Groq
            temperature=0.7,
            max_tokens=1024,
        )
        diary_content = response.choices[0].message.content
        
        # Save or update the generated diary to MongoDB
        db_client = get_db()
        db_client["daily_diaries"].update_one(
            {"user_id": user_id, "date": target_date},
            {"$set": {
                "user_id": user_id,
                "date": target_date,
                "pet_type": pet_type,
                "content": diary_content,
                "created_at": datetime.now()
            }},
            upsert=True
        )
        
        return diary_content
        
    except Exception as e:
        return f"일기 생성 중 오류가 발생했습니다: {str(e)}"

def get_diary_list(user_id: str, limit: int = 0) -> list:
    """
    Fetches the saved diaries for a user.
    If limit > 0, returns only that many recent diaries.
    """
    db_client = get_db()
    cursor = db_client["daily_diaries"].find({"user_id": user_id}).sort("date", -1)
    if limit > 0:
        cursor = cursor.limit(limit)
        
    diaries = []
    for doc in cursor:
        target_date = doc.get("date")
        video_url = ""
        
        if target_date:
            try:
                start_time = datetime.strptime(target_date, "%Y-%m-%d")
                end_time = start_time + timedelta(days=1)
                
                log_with_video = db_client["daily_logs"].find_one({
                    "user_id": user_id,
                    "timestamp": {"$gte": start_time, "$lt": end_time},
                    "video_url": {"$ne": ""}
                })
                if log_with_video and "video_url" in log_with_video:
                    video_url = log_with_video["video_url"].replace("minio:9000", "localhost:9000")
            except Exception:
                pass

        diaries.append({
            "date": target_date or "Unknown Date",
            "content": doc.get("content", ""),
            "pet_type": doc.get("pet_type", "Unknown"),
            "video_url": video_url
        })
    return diaries
