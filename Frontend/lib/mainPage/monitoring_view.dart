import 'package:flutter/material.dart';
import 'package:flutter_webrtc/flutter_webrtc.dart';
import 'package:pet_diary/services/webrtc_service.dart';
import 'package:pet_diary/config.dart';

class MonitoringView extends StatefulWidget {
  final String userId;

  const MonitoringView({super.key, required this.userId});

  @override
  State<MonitoringView> createState() => _MonitoringViewState();
}

class _MonitoringViewState extends State<MonitoringView> {
  final RTCVideoRenderer _remoteRenderer = RTCVideoRenderer();
  bool _isConnected = false;
  bool _isConnecting = true;
  WebRTCService? _webrtcService;

  @override
  void initState() {
    super.initState();
    _initRenderer();
  }

  Future<void> _initRenderer() async {
    await _remoteRenderer.initialize();
    
    // Connect to Signaling Server to receive stream
    _webrtcService = WebRTCService(
      userId: widget.userId,
      deviceId: 'viewer_${DateTime.now().millisecondsSinceEpoch}',
      signalingUrl: Config.signalingUrl,
      isSender: false,
    );

    _webrtcService?.onRemoteStream = (stream) {
      if (mounted) {
        setState(() {
          _remoteRenderer.srcObject = stream;
          _isConnected = true;
        });
      }
    };
    
    _webrtcService?.onConnectionState = (state) {
      debugPrint('WebRTC Viewer State: $state');
      if (mounted) {
        setState(() {
          _isConnected = state == RTCIceConnectionState.RTCIceConnectionStateConnected;
          if (_isConnected) _isConnecting = false;
        });
      }
    };

    await _webrtcService?.init(null);
    _webrtcService?.notifyViewerJoined();
    
    // 5초 후에도 연결되지 않으면 로딩 상태 해제 (안내 문구 표시용)
    Future.delayed(const Duration(seconds: 5), () {
      if (mounted && !_isConnected) {
        setState(() => _isConnecting = false);
      }
    });
  }

  @override
  void dispose() {
    _webrtcService?.dispose();
    _remoteRenderer.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      width: double.infinity,
      height: 220,
      decoration: BoxDecoration(
        color: Colors.black87,
        borderRadius: BorderRadius.circular(15),
        boxShadow: const [BoxShadow(color: Colors.black12, blurRadius: 4)],
      ),
      child: ClipRRect(
        borderRadius: BorderRadius.circular(15),
        child: Stack(
          children: [
            if (_isConnected)
              RTCVideoView(
                _remoteRenderer,
                objectFit: RTCVideoViewObjectFit.RTCVideoViewObjectFitCover,
              )
            else if (_isConnecting)
              const Center(
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    CircularProgressIndicator(color: Colors.orange),
                    SizedBox(height: 16),
                    Text(
                      '연결을 시도 중입니다...',
                      style: TextStyle(color: Colors.white70, fontSize: 13),
                    ),
                  ],
                ),
              )
            else
              const Center(
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Icon(Icons.videocam_off, color: Colors.white54, size: 40),
                    SizedBox(height: 12),
                    Text(
                      '연결된 CCTV가 없습니다.\n마이페이지에서 기기를 추가해주세요.',
                      textAlign: TextAlign.center,
                      style: TextStyle(color: Colors.white54, fontSize: 13),
                    ),
                  ],
                ),
              ),
            
            // 상태 표시
            Positioned(
              top: 12,
              right: 12,
              child: Container(
                padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                decoration: BoxDecoration(
                  color: _isConnected ? Colors.red : (_isConnecting ? Colors.orange : Colors.grey[700]),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Container(
                      width: 6,
                      height: 6,
                      decoration: const BoxDecoration(color: Colors.white, shape: BoxShape.circle),
                    ),
                    const SizedBox(width: 6),
                    Text(
                      _isConnected ? 'LIVE' : (_isConnecting ? 'CONNECTING' : 'OFFLINE'),
                      style: const TextStyle(color: Colors.white, fontSize: 10, fontWeight: FontWeight.bold),
                    ),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
