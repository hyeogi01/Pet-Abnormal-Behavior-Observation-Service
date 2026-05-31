from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
import json

router = APIRouter()

# 기기 페어링 정보 및 메타데이터 임시 저장
user_devices = {}
# 페어링 코드 -> 사용자 아이디 매핑 (임시)
pairing_codes = {}

class PairRequest(BaseModel):
    pairing_code: str
    device_model: str = "Unknown Device"

class RegisterCodeRequest(BaseModel):
    pairing_code: str
    user_id: str

class ReconnectRequest(BaseModel):
    device_id: str
    user_id: str
    device_model: str = "Unknown Device"

@router.post("/api/devices/register-code")
async def register_pairing_code(req: RegisterCodeRequest):
    pairing_codes[req.pairing_code] = req.user_id
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

@router.post("/api/devices/reconnect")
async def reconnect_device(req: ReconnectRequest):
    """공기계 앱 재시작 시 페어링 코드 없이 기존 device_id로 재등록"""
    if req.user_id not in user_devices:
        user_devices[req.user_id] = []
    user_devices[req.user_id] = [d for d in user_devices[req.user_id] if d['device_id'] != req.device_id]
    user_devices[req.user_id].append({
        "device_id": req.device_id,
        "model": req.device_model,
        "connected_at": "방금 전"
    })
    print(f"[WebRTC] Reconnected device {req.device_id} for user {req.user_id}")
    return {"status": "success", "device_id": req.device_id, "user_id": req.user_id}

@router.delete("/api/devices/{user_id}/{device_id}")
async def delete_device(user_id: str, device_id: str):
    if user_id in user_devices:
        original = user_devices[user_id]
        user_devices[user_id] = [d for d in original if d['device_id'] != device_id]
        if len(user_devices[user_id]) < len(original):
            return {"status": "success", "message": "기기가 삭제되었습니다."}
    return {"status": "error", "message": "기기를 찾을 수 없습니다."}

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

    async def relay_message(self, message: str, user_id: str, sender_device_id: str):
        """
        뷰어(viewer_*) → 특정 카메라로, 카메라 → 해당 카메라를 타겟으로 하는 뷰어들로 릴레이.
        뷰어 device_id 형식: viewer_{cam_device_id}_{timestamp}
        """
        if user_id not in self.active_connections:
            return

        if sender_device_id.startswith('viewer_'):
            # 뷰어 → 타겟 카메라로 전달
            # 예: 'viewer_cam_123456_1717000000000' → target: 'cam_123456'
            after_prefix = sender_device_id[len('viewer_'):]
            target_cam_id = after_prefix[:after_prefix.rfind('_')]
            target_ws = self.active_connections[user_id].get(target_cam_id)
            if target_ws:
                try:
                    await target_ws.send_text(message)
                except Exception as e:
                    print(f"Error sending to cam {target_cam_id}: {e}")
        else:
            # 카메라 → 이 카메라를 타겟으로 하는 뷰어들에게 전달
            prefix = f'viewer_{sender_device_id}_'
            for d_id, connection in list(self.active_connections[user_id].items()):
                if d_id.startswith(prefix):
                    try:
                        await connection.send_text(message)
                    except Exception as e:
                        print(f"Error sending to viewer {d_id}: {e}")

manager = ConnectionManager()

@router.websocket("/ws/webrtc/{user_id}/{device_id}")
async def webrtc_endpoint(websocket: WebSocket, user_id: str, device_id: str):
    await manager.connect(websocket, user_id, device_id)
    try:
        while True:
            data = await websocket.receive_text()
            # 수신한 시그널링 데이터(SDP, ICE)를 릴레이
            await manager.relay_message(data, user_id, device_id)
    except WebSocketDisconnect:
        manager.disconnect(websocket, user_id, device_id)

