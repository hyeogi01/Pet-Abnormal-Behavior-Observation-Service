from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
import json

import os

router = APIRouter()

# 기기 정보 및 페어링 코드 영구 저장을 위한 파일 경로
DEVICES_FILE = "devices_data.json"

def load_data():
    if os.path.exists(DEVICES_FILE):
        try:
            with open(DEVICES_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            pass
    return {"user_devices": {}, "pairing_codes": {}}

def save_data(data):
    with open(DEVICES_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# 데이터 로드
initial_data = load_data()
user_devices = initial_data["user_devices"]
pairing_codes = initial_data["pairing_codes"]

class PairRequest(BaseModel):
    pairing_code: str
    device_model: str = "Unknown Device"

class RegisterCodeRequest(BaseModel):
    pairing_code: str
    user_id: str

@router.post("/api/devices/register-code")
async def register_pairing_code(req: RegisterCodeRequest):
    pairing_codes[req.pairing_code] = req.user_id
    save_data({"user_devices": user_devices, "pairing_codes": pairing_codes})
    print(f"[WebRTC] Registered code {req.pairing_code} for user {req.user_id}")
    return {"status": "success"}

@router.post("/api/devices/pair")
async def pair_device(req: PairRequest):
    code = req.pairing_code
    
    # 등록된 페어링 코드로 user_id 조회
    user_id = pairing_codes.get(code)
    
    if not user_id:
        # 테스트를 위해 코드가 000000 이면 test_user로 허용 (디버깅용)
        if code == "000000":
            user_id = "test_user"
        else:
            return {"status": "error", "message": "유효하지 않은 연결 코드입니다."}
    
    device_id = f"cam_{code}"
    
    if user_id not in user_devices:
        user_devices[user_id] = []
    
    # 동일한 디바이스 ID가 있으면 덮어쓰기
    user_devices[user_id] = [d for d in user_devices[user_id] if d['device_id'] != device_id]
    user_devices[user_id].append({
        "device_id": device_id,
        "model": req.device_model,
        "connected_at": "방금 전"
    })
    
    save_data({"user_devices": user_devices, "pairing_codes": pairing_codes})
    
    return {
        "status": "success",
        "device_id": device_id,
        "user_id": user_id,
        "message": "페어링 성공"
    }

@router.get("/api/devices/list/{user_id}")
async def list_devices(user_id: str):
    devices = user_devices.get(user_id, [])
    return {
        "status": "success",
        "data": devices
    }

@router.delete("/api/devices/{user_id}/{device_id}")
async def delete_device(user_id: str, device_id: str):
    if user_id in user_devices:
        user_devices[user_id] = [d for d in user_devices[user_id] if d['device_id'] != device_id]
        save_data({"user_devices": user_devices, "pairing_codes": pairing_codes})
    return {"status": "success"}

# WebRTC 시그널링을 위한 WebSocket 연결 관리
class ConnectionManager:
    def __init__(self):
        # user_id: { device_id: WebSocket }
        self.active_connections = {}

    async def connect(self, websocket: WebSocket, user_id: str, device_id: str):
        await websocket.accept()
        if user_id not in self.active_connections:
            self.active_connections[user_id] = {}
        self.active_connections[user_id][device_id] = websocket
        print(f"[WebRTC] {device_id} connected for user {user_id}")

    def disconnect(self, websocket: WebSocket, user_id: str, device_id: str):
        if user_id in self.active_connections and device_id in self.active_connections[user_id]:
            del self.active_connections[user_id][device_id]
            print(f"[WebRTC] {device_id} disconnected for user {user_id}")

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def send_to_device(self, message: str, user_id: str, target_device_id: str):
        if user_id in self.active_connections and target_device_id in self.active_connections[user_id]:
            try:
                await self.active_connections[user_id][target_device_id].send_text(message)
            except Exception as e:
                print(f"Error sending to {target_device_id}: {e}")

    async def broadcast_to_user(self, message: str, user_id: str, sender_device_id: str):
        # 같은 유저 아이디로 접속한 다른 모든 기기에게 메시지 (Offer/Answer/ICE) 전달
        if user_id in self.active_connections:
            for d_id, connection in self.active_connections[user_id].items():
                if d_id != sender_device_id:
                    try:
                        await connection.send_text(message)
                    except Exception as e:
                        print(f"Error sending to {d_id}: {e}")

manager = ConnectionManager()

@router.websocket("/ws/webrtc/{user_id}/{device_id}")
async def webrtc_endpoint(websocket: WebSocket, user_id: str, device_id: str):
    await manager.connect(websocket, user_id, device_id)
    try:
        while True:
            data = await websocket.receive_text()
            
            # 수신한 시그널링 데이터(SDP, ICE)를 타겟이 있으면 타겟에게, 없으면 모두에게 릴레이
            try:
                json_data = json.loads(data)
                target = json_data.get("target")
                
                # 메시지에 발신자 정보 추가
                json_data["sender"] = device_id
                data_with_sender = json.dumps(json_data)
                
                if target:
                    await manager.send_to_device(data_with_sender, user_id, target)
                else:
                    await manager.broadcast_to_user(data_with_sender, user_id, device_id)
            except json.JSONDecodeError:
                await manager.broadcast_to_user(data, user_id, device_id)
                
    except WebSocketDisconnect:
        manager.disconnect(websocket, user_id, device_id)
