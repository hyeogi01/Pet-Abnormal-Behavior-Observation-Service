import 'package:flutter/material.dart';
import 'package:flutter_webrtc/flutter_webrtc.dart';
import 'package:pet_diary/services/webrtc_service.dart';
import 'package:pet_diary/config.dart';

/// 공기계(송출 기기)에서 실제로 영상을 송출하는 화면
class CamSenderPage extends StatefulWidget {
  final String deviceId;
  final String userId;

  const CamSenderPage({
    super.key,
    required this.deviceId,
    required this.userId,
  });

  @override
  State<CamSenderPage> createState() => _CamSenderPageState();
}

class _CamSenderPageState extends State<CamSenderPage> {
  final RTCVideoRenderer _localRenderer = RTCVideoRenderer();
  MediaStream? _localStream;
  bool _isStreaming = false;
  WebRTCService? _webrtcService;

  @override
  void initState() {
    super.initState();
    _initRenderer();
  }

  Future<void> _initRenderer() async {
    await _localRenderer.initialize();
    _startCamera();
  }

  Future<void> _startCamera() async {
    final Map<String, dynamic> mediaConstraints = {
      'audio': true,
      'video': {
        'facingMode': 'environment', // 후면 카메라 기본
      }
    };

    try {
      final stream = await navigator.mediaDevices.getUserMedia(mediaConstraints);
      _localRenderer.srcObject = stream;
      setState(() {
        _localStream = stream;
        _isStreaming = true;
      });
      
      // WebRTC Signaling 연결 및 스트림 송출 시작
      _webrtcService = WebRTCService(
        userId: widget.userId,
        deviceId: widget.deviceId,
        signalingUrl: Config.signalingUrl,
        isSender: true,
      );
      
      _webrtcService?.onConnectionState = (state) {
        debugPrint('WebRTC Sender State: $state');
      };
      
      await _webrtcService?.init(_localStream);
      
    } catch (e) {
      debugPrint('Camera/WebRTC Error: $e');
      setState(() {
        _isStreaming = false;
      });
      if (mounted) {
        String errorMsg = '카메라를 시작할 수 없습니다. 하드웨어나 권한을 확인해주세요.';
        if (e.toString().contains('DevicesNotFoundError')) {
          errorMsg = '연결된 카메라가 없습니다. (테스트용 가상 송출을 시도합니다)';
          // TODO: 카메라가 없을 때 더미 영상을 송출하는 로직을 추가할 수 있습니다.
        }
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text(errorMsg),
            backgroundColor: Colors.redAccent,
            duration: const Duration(seconds: 5),
          ),
        );
      }
    }
  }

  @override
  void dispose() {
    _webrtcService?.dispose();
    _localStream?.dispose();
    _localRenderer.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      body: Stack(
        children: [
          // 영상 렌더러
          Positioned.fill(
            child: _isStreaming
                ? RTCVideoView(
                    _localRenderer,
                    objectFit: RTCVideoViewObjectFit.RTCVideoViewObjectFitCover,
                  )
                : const Center(
                    child: CircularProgressIndicator(color: Colors.orange),
                  ),
          ),
          
          // 상단 UI (뒤로가기 및 상태)
          SafeArea(
            child: Padding(
              padding: const EdgeInsets.all(16.0),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Container(
                    decoration: BoxDecoration(
                      color: Colors.black45,
                      borderRadius: BorderRadius.circular(30),
                    ),
                    child: IconButton(
                      icon: const Icon(Icons.close, color: Colors.white),
                      onPressed: () {
                        // 송출 종료 시 재확인 다이얼로그 띄울 수 있음
                        Navigator.pop(context);
                      },
                    ),
                  ),
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                    decoration: BoxDecoration(
                      color: Colors.black45,
                      borderRadius: BorderRadius.circular(20),
                    ),
                    child: Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Container(
                          width: 8,
                          height: 8,
                          decoration: const BoxDecoration(
                            color: Colors.red,
                            shape: BoxShape.circle,
                          ),
                        ),
                        const SizedBox(width: 8),
                        const Text(
                          'LIVE 송출 중',
                          style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            ),
          ),
          
          // 하단 정보 표시
          Positioned(
            bottom: 40,
            left: 20,
            right: 20,
            child: Container(
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: Colors.black54,
                borderRadius: BorderRadius.circular(15),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                mainAxisSize: MainAxisSize.min,
                children: [
                  const Text(
                    '이 기기는 CCTV 모드로 동작 중입니다.',
                    style: TextStyle(color: Colors.white, fontSize: 16, fontWeight: FontWeight.bold),
                  ),
                  const SizedBox(height: 8),
                  Text(
                    'Device ID: ${widget.deviceId}',
                    style: const TextStyle(color: Colors.white70, fontSize: 12),
                  ),
                  const SizedBox(height: 4),
                  const Text(
                    '메인 기기에서 영상을 확인할 수 있습니다.\n배터리 소모가 크므로 전원을 연결해 두는 것을 권장합니다.',
                    style: TextStyle(color: Colors.white70, fontSize: 12, height: 1.5),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}
