import os
import json
from collections import Counter
from datetime import datetime, timedelta
from groq import Groq
from firebase_admin import db as firebase_db
from FastAPI.main.db import MINIO_PUBLIC_URL

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

# ── 감정 → 자연어 힌트 (LLM이 자연스러운 문장을 쓸 수 있도록 유도) ──
EMOTION_MOOD_HINT = {
    "dog_happy":    "매우 신나고 기분 좋은",
    "dog_relaxed":  "편안하고 차분한",
    "dog_anxious":  "불안하고 축 처진",
    "dog_sad":      "슬프고 기운 없는",
    "dog_angry":    "화나고 기분 나쁜",
    "dog_confused": "헷갈리고 이상한 느낌의",
    "cat_happy":    "기분 좋고 즐거운",
    "cat_relaxed":  "느긋하고 편안한",
    "cat_attentive":"뭔가 궁금한 것이 많은",
    "cat_sad":      "슬프고 기운 없는",
}

# ── 행동 코드 → 한국어 ──
BEHAVIOR_KR = {
    "DOG_BODYLOWER":  "몸을 낮추기",
    "DOG_BODYSCRATCH":"몸 긁기",
    "DOG_BODYSHAKE":  "몸 흔들기",
    "DOG_FEETUP":     "앞발 들기",
    "DOG_FOOTUP":     "발 들기",
    "DOG_HEADING":    "고개 두리번거리기",
    "DOG_LYING":      "누워있기",
    "DOG_MOUNTING":   "올라타기",
    "DOG_SIT":        "앉아있기",
    "DOG_TAILING":    "꼬리 흔들기",
    "DOG_TAILLOW":    "꼬리 내리기",
    "DOG_TURN":       "빙빙 돌기",
    "DOG_WALKRUN":    "뛰어다니기",
    "CAT_ARCH":       "등 구부리기",
    "CAT_ARMSTRETCH": "앞발 쭉 뻗기",
    "CAT_FOOTPUSH":   "발 꾹꾹이",
    "CAT_GETDOWN":    "내려오기",
    "CAT_GROOMING":   "그루밍하기",
    "CAT_HEADING":    "고개 두리번거리기",
    "CAT_LAYDOWN":    "엎드리기",
    "CAT_LYING":      "누워있기",
    "CAT_ROLL":       "뒹굴기",
    "CAT_SITDOWN":    "앉아있기",
    "CAT_TAILING":    "꼬리 움직이기",
    "CAT_WALKRUN":    "뛰어다니기",
}

# ── 사운드 코드 → 한국어 ──
SOUND_KR = {
    "dog_bark":             "짖기",
    "dog_howling":          "하울링",
    "dog_respiratory_event":"호흡 이상 소리",
    "dog_whining":          "낑낑거리기",
    "cat_aggressive":       "하악질",
    "cat_positive":         "그루밍 소리",
}

# ── 부정적으로 간주하는 소리/행동 ──
NEGATIVE_SOUNDS    = {"dog_bark", "dog_howling", "dog_whining", "cat_aggressive"}
NEGATIVE_BEHAVIORS = {"DOG_BODYLOWER", "DOG_TAILLOW", "DOG_MOUNTING"}


_POSITIVE_EMOTIONS = {"dog_happy", "dog_relaxed", "cat_happy", "cat_relaxed", "cat_attentive"}

def _build_summary_card(logs: list) -> tuple[str, bool, bool, bool, bool]:
    """
    로그를 집계해 LLM에 넘길 요약 카드 문자열과 플래그를 반환한다.
    Returns: (summary_text, has_negative_sound, has_negative_behavior, has_patella, dominant_positive)
    dominant_positive: 전체 감정 중 긍정 감정이 절반 이상일 때 True
    """
    emotion_counter  = Counter()
    behavior_counter = Counter()
    sound_counter    = Counter()
    patella_issues   = 0

    _skip = {"Unknown", "None", "background", "unknown", "none", ""}

    for log in logs:
        result = log.get("analysis_result", {})
        if result.get("status") != "success":
            continue

        emo = result.get("behavior_analysis", {}).get("emotion", "")
        if emo and emo not in _skip:
            emotion_counter[emo] += 1

        beh = result.get("behavior_analysis", {}).get("detected_behavior", "")
        if beh and beh not in _skip:
            behavior_counter[beh] += 1

        snd = result.get("audio_analysis", {}).get("detected_sound", "")
        if snd and snd not in _skip:
            sound_counter[snd] += 1

        patella = result.get("patella_analysis", {}).get("status", "normal")
        patella_str = str(patella).lower() if patella is not None else "normal"
        if patella_str not in ("normal", "unknown", ""):
            patella_issues += 1

    lines = []

    if emotion_counter:
        parts = []
        for emo, cnt in emotion_counter.most_common(3):
            hint = EMOTION_MOOD_HINT.get(emo)
            if hint:
                parts.append(f"{hint} 감정 ({cnt}회)")
        if parts:
            lines.append(f"주된 감정: {', '.join(parts)}")

    if behavior_counter:
        parts = [f"{BEHAVIOR_KR.get(b, b)} ({c}회)" for b, c in behavior_counter.most_common(3)]
        lines.append(f"주요 행동: {', '.join(parts)}")

    if sound_counter:
        parts = [f"{SOUND_KR.get(s, s)} ({c}회)" for s, c in sound_counter.most_common()]
        lines.append(f"특이 소리: {', '.join(parts)}")

    if patella_issues:
        lines.append(f"슬개골 이상: {patella_issues}회 감지됨")

    has_negative_sound    = bool(sound_counter.keys() & NEGATIVE_SOUNDS)
    has_negative_behavior = bool(behavior_counter.keys() & NEGATIVE_BEHAVIORS)
    has_patella           = patella_issues > 0

    total_emotions    = sum(emotion_counter.values())
    positive_count    = sum(cnt for emo, cnt in emotion_counter.items() if emo in _POSITIVE_EMOTIONS)
    dominant_positive = (positive_count >= total_emotions / 2) if total_emotions > 0 else True

    return "\n".join(lines), has_negative_sound, has_negative_behavior, has_patella, dominant_positive

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

    # ─── 전체 로그 집계 → 요약 카드 생성 ───
    summary_card, has_negative_sound, has_negative_behavior, has_patella, dominant_positive = _build_summary_card(logs)

    if not summary_card.strip():
        return f"{target_date if target_date else '오늘'}의 분석 데이터가 충분하지 않아 일기를 작성하기 어렵습니다."

    pet_nickname = pet_info.get("pet_name", "나")

    # ─── 상황별 스토리 흐름 지시문 ───
    has_negative = has_negative_sound or has_negative_behavior
    sound_extra  = "소리도 냈으면 '큰 소리를 냈어요'처럼 짧게 덧붙여줘.\n" if has_negative_sound else ""

    if dominant_positive and has_negative and has_patella:
        # 긍정 majority + 부정 행동/소리 있음 + 슬개골
        story_instruction = (
            "①오늘 하루 기분을 밝게 한 문장으로 시작해줘. (예: '오늘은 기분 좋은 하루였어요!')\n"
            f"②가끔 기분이 좋지 않아서 축 처져있기도 했고, 다리도 조금 불편했다고 자연스럽게 이어줘. {sound_extra}"
            "③그래도 주인님을 보니 기운이 났다는 밝은 반전을 넣어줘.\n"
            "④걱정 끼쳐드려서 죄송하다고 짧게 사과하고, '사랑해요~' 같은 마무리로 끝내줘."
        )
    elif dominant_positive and has_negative:
        # 긍정 majority + 부정 행동/소리 있음
        story_instruction = (
            "①오늘 하루 기분을 밝게 한 문장으로 시작해줘. (예: '오늘은 기분 좋은 하루였어요!')\n"
            f"②근데 가끔 기분이 좋지 않아서 축 처져있기도 했다고 자연스럽게 이어줘. {sound_extra}"
            "③그래도 주인님을 보면 기운이 난다는 한 문장을 넣어줘.\n"
            "④걱정 끼쳐드려서 죄송하다고 짧게 사과하고, '내일도 기운 내볼게요! 사랑해요~' 같은 마무리로 끝내줘."
        )
    elif dominant_positive and has_patella:
        # 긍정 majority + 슬개골
        story_instruction = (
            "①오늘 하루 기분을 밝게 한 문장으로 시작해줘.\n"
            "②근데 다리가 조금 불편한 느낌이었다고 자연스럽게 이어줘.\n"
            "③그래도 주인님을 보면 기운이 난다는 한 문장을 넣어줘.\n"
            "④'사랑해요~' 같은 애정 표현으로 마무리해줘."
        )
    elif not dominant_positive and has_negative and has_patella:
        # 부정 majority + 부정 행동/소리 + 슬개골
        story_instruction = (
            "①오늘 하루 기분을 부드럽게 한 문장으로 시작해줘. (예: '조금 기운이 없는 하루였어요')\n"
            f"②왠지 다리가 조금 불편해서 기분이 좋지 않아 축 처져있었다고 이유를 자연스럽게 이어줘. {sound_extra}"
            "③그래도 주인님을 보니 기운이 나는 것 같았다는 밝은 반전을 한 문장으로 넣어줘.\n"
            "④시무룩하게 있으면서 걱정 끼쳐드려서 죄송하다고 짧게 사과해줘.\n"
            "⑤'내일은 조금 더 기운 내볼게요! 사랑해요~' 같은 밝은 마무리로 끝내줘."
        )
    elif not dominant_positive and has_negative:
        # 부정 majority + 부정 행동/소리
        story_instruction = (
            "①오늘 하루 기분을 부드럽게 한 문장으로 시작해줘. (예: '왠지 마음이 편하지 않은 하루였어요')\n"
            f"②왠지 기분이 좋지 않아서 축 처져있었다고 이유를 자연스럽게 이어줘. {sound_extra}"
            "③그래도 주인님을 보니 기운이 나는 것 같았다는 밝은 반전을 한 문장으로 넣어줘.\n"
            "④걱정 끼쳐드려서 죄송하다고 짧게 사과해줘.\n"
            "⑤'내일은 조금 더 기운 내볼게요! 사랑해요~' 같은 밝은 마무리로 끝내줘."
        )
    elif has_patella:
        # 긍정/중립 + 슬개골만
        story_instruction = (
            "①오늘 하루 기분을 부드럽게 한 문장으로 시작해줘.\n"
            "②왠지 다리가 조금 불편해서 그랬던 것 같다고 이유를 자연스럽게 이어줘.\n"
            "③그래도 주인님을 보니 기운이 나는 것 같았다는 밝은 반전을 한 문장으로 넣어줘.\n"
            "④'사랑해요~' 같은 애정 표현으로 밝게 마무리해줘."
        )
    else:
        # 순수 긍정
        story_instruction = (
            "①오늘 하루 기분을 밝고 신나게 한 문장으로 시작해줘.\n"
            "②즐거웠던 것 하나를 자연스럽게 이어줘.\n"
            "③'주인님 보고 싶어요! 사랑해요~' 같은 애정 표현으로 마무리해줘."
        )

    # ─────────────────────────── Generate Pet Diary (1st Person) ───────────────────────────
    pet_diary_prompt = f"""
[오늘의 요약 카드] ← 이 내용만 사용해서 일기를 써줘
{summary_card}

[절대 쓰면 안 되는 것]
- 요약 카드에 없는 상황, 대화, 소리, 사건
- 주인님이 한 말이나 행동 (주인님 시점 묘사 금지)

[스토리 흐름]
{story_instruction}

흐름이 자연스럽게 이어지도록 한 문단으로 써줘. 같은 내용 두 번 쓰지 마.

[참고 예시 - 기운 없는 날 + 다리 불편]
오늘은 조금 기운이 없는 하루였어요. 왠지 다리가 조금 불편해서 그랬던 것 같아요. 그래도 주인님을 보니까 기운이 나는 것 같았어요! 시무룩하게 있으면서 걱정 끼쳐드려서 죄송해요. 내일은 조금 더 기운 내볼게요! 사랑해요~

[참고 예시 - 즐거운 날]
오늘은 진짜진짜 신나는 하루였어요! 뛰어다니고 앞발도 들고 너무 즐거웠어요. 주인님 빨리 오세요, 보고 싶어요~
    """

    # ─────────────────────────── Generate AI Report (Professional 1-liner) ───────────────────────────
    report_prompt = f"""
당신은 전문 반려동물 행동 분석가입니다.
아래의 오늘 하루 분석 요약을 바탕으로, 보호자에게 줄 'AI 행동 분석 레포트'를 작성하세요.

[오늘의 요약]
{summary_card}

[작성 지침]
1. 반드시 딱 한 줄(한 문장)로만 작성하세요.
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
                        f"너는 5살짜리 {pet_type} '{pet_nickname}'이야. 주인님께 오늘 하루 일기를 쓰고 있어.\n"
                        "규칙:\n"
                        "1. 반드시 한국어로만 써. 영어 단어는 단 하나도 쓰면 안 돼.\n"
                        "2. 자신을 가리킬 때는 반드시 '저'를 사용해. '나'는 절대 쓰지 마.\n"
                        "3. 보호자는 반드시 '주인님'으로 불러. '주인'만 단독으로 쓰지 마.\n"
                        "4. 문장 끝은 반드시 '~요'로 끝내. '~어', '~다', '~야'로 끝나면 안 돼.\n"
                        "5. 짧고 단순한 문장으로, 또박또박 쓰는 밝고 발랄한 아이처럼 써.\n"
                        "6. 요약 카드에 있는 내용만 써. 없는 상황이나 대화는 절대 만들어내면 안 돼.\n"
                        "7. '슬개골', 'AI 탐지', 'Normal', '슬개골 탈구', 'confidence' 같은 전문 용어는 절대 쓰지 마.\n"
                        "8. 감정이 여러 개면 갑자기 전환하지 말고 자연스럽게 이어줘. "
                        "(예: 처음엔 좀 불안했는데요, 나중엔 괜찮아졌어요)\n"
                        "9. '주인님한테'가 아니라 '주인님께' 또는 '주인님이'로 써줘.\n"
                        "10. 행동을 직접 묘사하지 말고 감정이나 느낌으로 표현해줘. "
                        "(예: '몸을 낮췄어요' → '왠지 기분이 좋지 않아서 축 처져있었어요')\n"
                        "11. 부정적인 감정도 강한 단어 없이 부드럽게 돌려서 표현해줘. "
                        "(예: '무서운 하루' → '조금 기운이 없는 하루', '불안하고 무서웠어요' → '왠지 마음이 편하지 않았어요')\n"
                        "12. 같은 내용을 두 번 반복하지 마. 다리 불편함도 마지막에 딱 한 번만 써.\n"
                        "13. 한국어와 한글만 사용해. 한자, 중국어, 일본어 문자는 절대 쓰지 마.\n"
                        "14. 150자 이내로 짧고 자연스럽게 써."
                    )
                },
                {"role": "user", "content": pet_diary_prompt}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.7,
            max_tokens=250,
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
            model="llama-3.3-70b-versatile",
            temperature=0.4,
            max_tokens=80,
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
                    url = log["image_url"]
                    # Normalize legacy internal URLs to the current public URL
                    url = url.replace("http://localhost:9000", MINIO_PUBLIC_URL)
                    url = url.replace("http://minio:9000", MINIO_PUBLIC_URL)
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
