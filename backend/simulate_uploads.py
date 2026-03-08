import os
import cv2
import requests
import time
import datetime

# --- CONFIGURATION ---
VIDEO_PATH = "sample1.mp4" 
CLIP_NAME = "sample_clip_15s.mp4"
API_URL = "http://localhost:8000/api/daily-behavior"
USER_ID = "hello"  
PET_TYPE = "dog"   
NUM_CHUNKS = 24    # Number of simulation uploads (4 days * 6 chunks per day)
# ---------------------

def prepare_15s_clip():
    """Extracts only the first 15 seconds of the video to drastically reduce processing time."""
    if os.path.exists(CLIP_NAME):
        print(f"Using existing clip: {CLIP_NAME}")
        return True

    print(f"Opening video: {VIDEO_PATH} to extract 15s clip...")
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open {VIDEO_PATH}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    frames_to_read = int(fps * 15)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(CLIP_NAME, fourcc, fps, (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ))

    print(f"Saving ~{frames_to_read} frames to {CLIP_NAME}...")
    for _ in range(frames_to_read):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        
    out.release()
    cap.release()
    print("Clip extraction complete.")
    return True

def simulate_historical_uploads():
    success = prepare_15s_clip()
    if not success:
        return
        
    print(f"\nStarting {NUM_CHUNKS} uploads utilizing {CLIP_NAME}...")

    # For 24 chunks spanning 4 days: 6 chunks per day
    for chunk_idx in range(NUM_CHUNKS):
        # Calculate historical time (4 days back to today)
        days_ago = 3 - (chunk_idx // 6)
        chunk_time = datetime.datetime.now() - datetime.timedelta(days=days_ago)
        
        # Space them 2 hours apart starting from 10:00 AM
        hour_offset = 10 + (chunk_idx % 6) * 2
        chunk_time = chunk_time.replace(hour=hour_offset if hour_offset < 24 else 23, minute=0, second=0, microsecond=0)

        print(f"[{chunk_idx+1}/{NUM_CHUNKS}] Uploading log for timestamp: {chunk_time.strftime('%Y-%m-%d %H:%M:%S')}")

        try:
            with open(CLIP_NAME, 'rb') as f:
                files = {'file': (CLIP_NAME, f, 'video/mp4')}
                data = {
                    'user_id': USER_ID,
                    'pet_type': PET_TYPE,
                    'timestamp': chunk_time.isoformat()
                }
                
                response = requests.post(API_URL, data=data, files=files)
                
                if response.status_code == 200:
                    resp_data = response.json()
                    status = resp_data.get('status')
                    if status == "success":
                        ai_info = resp_data.get('ai_inference', {}).get('summary', '')
                        print(f" -> AI Result: {ai_info}")
                    else:
                        print(f" -> Backend Warning: {resp_data.get('message')}")
                else:
                    print(f" -> Upload Failed: HTTP {response.status_code}")
                    print(response.text)
                    
        except Exception as e:
            print(f"Error uploading chunk {chunk_idx}: {e}")
            
        time.sleep(1) # tiny delay to let the backend breathe
        
    print("\n✅ Simulation complete! You can now check the Flutter app photo gallery.")

if __name__ == "__main__":
    simulate_historical_uploads()
