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

  void _showFullScreen() {
    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (context) => Scaffold(
          backgroundColor: Colors.black,
          body: Stack(
            children: [
              Positioned.fill(
                child: RTCVideoView(
                  _remoteRenderer,
                  objectFit: RTCVideoViewObjectFit.RTCVideoViewObjectFitCover,
                ),
              ),
              SafeArea(
                child: Align(
                  alignment: Alignment.topRight,
                  child: Padding(
                    padding: const EdgeInsets.all(16.0),
                    child: IconButton(
                      icon: const Icon(Icons.close, color: Colors.white, size: 32),
                      onPressed: () => Navigator.pop(context),
                    ),
                  ),
                ),
              ),
              // 상태 표시 (전체화면용)
              SafeArea(
                child: Align(
                  alignment: Alignment.topLeft,
                  child: Padding(
                    padding: const EdgeInsets.all(24.0),
                    child: Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Container(
                          width: 8,
                          height: 8,
                          decoration: const BoxDecoration(color: Colors.red, shape: BoxShape.circle),
                        ),
                        const SizedBox(width: 8),
                        const Text('LIVE 송출 중', style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold)),
                      ],
                    ),
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    ).then((_) {
      if (mounted) setState(() {});
    });
  }

  @override
  Widget build(BuildContext context) {
    return AspectRatio(
      aspectRatio: 16 / 9,
      child: Container(
        width: double.infinity,
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
              Positioned.fill(
                child: RTCVideoView(
                  _remoteRenderer,
                  objectFit: RTCVideoViewObjectFit.RTCVideoViewObjectFitCover,
                ),
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
            
            // 전체화면 확대 버튼
            if (_isConnected)
              Positioned(
                bottom: 8,
                right: 8,
                child: IconButton(
                  icon: const Icon(Icons.fullscreen, color: Colors.white, size: 28),
                  onPressed: _showFullScreen,
                  style: IconButton.styleFrom(
                    backgroundColor: Colors.black45,
                    padding: const EdgeInsets.all(8),
                  ),
                ),
              ),
          ],
        ),
      ),
    ));
  }
}
