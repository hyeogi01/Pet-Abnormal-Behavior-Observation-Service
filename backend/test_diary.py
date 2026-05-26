"""
일기 프롬프트 테스트 스크립트
사용법: python backend/test_diary.py
"""
import os
import sys

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# ── Firebase 초기화 ──
import firebase_admin
from firebase_admin import credentials

_KEY_PATH = os.path.join(os.path.dirname(__file__), 'FastAPI', 'key', 'testApi.json')
if not firebase_admin._apps:
    cred = credentials.Certificate(os.path.abspath(_KEY_PATH))
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://test-25cac-default-rtdb.asia-southeast1.firebasedatabase.app/'
    })

# ── 테스트 대상 ──
from FastAPI.main.llm_diary import (
    get_daily_logs_for_diary,
    _build_summary_card,
    generate_daily_diary,
)

USER_ID     = "66"
TARGET_DATE = "2026-03-30"

def main():
    print(f"=== 로그 조회: user={USER_ID}, date={TARGET_DATE} ===")
    logs = get_daily_logs_for_diary(USER_ID, TARGET_DATE)
    print(f"총 {len(logs)}개 로그 조회됨\n")

    if not logs:
        print("로그가 없습니다.")
        return

    print("=== 요약 카드 (LLM에 전달되는 데이터) ===")
    summary_card, has_negative, has_patella = _build_summary_card(logs)
    print(summary_card)
    print(f"\n▶ has_negative={has_negative}, has_patella={has_patella}\n")

    print("=" * 50)
    ans = input("Groq API를 호출해서 실제 일기를 생성할까요? (y/n): ").strip().lower()
    if ans == "y":
        print("\n=== 생성된 일기 ===")
        diary = generate_daily_diary(USER_ID, TARGET_DATE)
        print(diary)
    else:
        print("일기 생성 건너뜀.")

if __name__ == "__main__":
    main()
