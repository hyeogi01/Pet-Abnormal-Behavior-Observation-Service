import 'package:flutter/material.dart';
import 'package:flutter_webrtc/flutter_webrtc.dart';
import 'dart:convert';
import 'package:http/http.dart' as http;
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
  
  List<dynamic> _devices = [];
  String? _selectedDeviceId;
  bool _isRecording = false;
  bool _hasVoiceStream = false;
  bool _isMuted = false;
  MediaStream? _localAudioStream;
  String? _wsError;
  String _statusMsg = '연결을 시도 중입니다...';

  @override
  void initState() {
    super.initState();
    _fetchDevices();
  }

  Future<void> _fetchDevices() async {
    try {
      final response = await http.get(
        Uri.parse('${Config.apiBaseUrl}/api/devices/list/${widget.userId}'),
        headers: Config.ngrokHeaders,
      );
      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        if (data['status'] == 'success') {
          if (mounted) {
            setState(() {
              _devices = data['data'];
              if (_devices.isNotEmpty) {
                _selectedDeviceId = _devices[0]['device_id'];
              }
            });
          }
        }
      }
    } catch (e) {
      debugPrint('Failed to fetch devices: $e');
    } finally {
      _initRenderer();
    }
  }

  Future<void> _initRenderer() async {
    if (_selectedDeviceId == null) {
      if (mounted) {
        setState(() => _isConnecting = false);
      }
      return;
    }
    
    await _remoteRenderer.initialize();
    _remoteRenderer.muted = false; // 공기계 오디오 수신 재생 활성화

    // Connect to Signaling Server to receive stream
    _webrtcService = WebRTCService(
      userId: widget.userId,
      deviceId: 'viewer_${DateTime.now().millisecondsSinceEpoch}',
      signalingUrl: Config.signalingUrl,
      isSender: false,
      targetDeviceId: _selectedDeviceId,
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

    _webrtcService?.onError = (error) {
      debugPrint('WebRTC Error: $error');
      if (mounted) {
        setState(() {
          _wsError = error;
          _isConnecting = false;
        });
      }
    };

    _webrtcService?.onStatus = (status) {
      if (mounted) {
        setState(() => _statusMsg = status);
      }
    };

    await _webrtcService?.init(null);
    
    // 5초 후에도 연결되지 않으면 로딩 상태 해제 (안내 문구 표시용)
    Future.delayed(const Duration(seconds: 5), () {
      if (mounted && !_isConnected) {
        setState(() => _isConnecting = false);
      }
    });
  }

  @override
  void dispose() {
    _localAudioStream?.getTracks().forEach((track) => track.stop());
    _localAudioStream?.dispose();
    _webrtcService?.dispose();
    _remoteRenderer.dispose();
    super.dispose();
  }
  
  Future<void> _changeDevice(String? newDeviceId) async {
    if (newDeviceId == null || newDeviceId == _selectedDeviceId) return;
    
    // 기존 연결 종료
    _webrtcService?.dispose();
    _localAudioStream?.getTracks().forEach((track) => track.stop());
    _localAudioStream?.dispose();
    
    setState(() {
      _selectedDeviceId = newDeviceId;
      _isConnected = false;
      _isConnecting = true;
      _isRecording = false;
      _hasVoiceStream = false;
    });
    
    await _initRenderer();
  }

  Future<void> _toggleRecording() async {
    if (_isRecording) {
      // 녹음 중지 (스트림 유지하되 상태만 변경하거나, 실제 전송 준비)
      setState(() {
        _isRecording = false;
        _hasVoiceStream = true;
      });
    } else {
      // 녹음 시작 (마이크 권한 요청 및 스트림 획득)
      try {
        final stream = await navigator.mediaDevices.getUserMedia({
          'audio': {
            'echoCancellation': true,
            'noiseSuppression': true,
          }
        });
        _localAudioStream = stream;
        setState(() {
          _isRecording = true;
          _hasVoiceStream = false;
        });
      } catch (e) {
        debugPrint('Microphone error: $e');
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(content: Text('마이크 권한을 확인해주세요.')),
          );
        }
      }
    }
  }

  Future<void> _sendVoice() async {
    if (_localAudioStream != null) {
      await _webrtcService?.addAudioTrack(_localAudioStream!);
      // _hasVoiceStream = false; // 버튼을 유지하기 위해 주석 처리 또는 제거
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('음성이 기기로 송출되었습니다.')),
        );
      }
    }
  }

  void _deleteVoice() {
    _localAudioStream?.getTracks().forEach((track) => track.stop());
    _localAudioStream?.dispose();
    _localAudioStream = null;
    setState(() {
      _hasVoiceStream = false;
    });
  }

  void _toggleMute() {
    setState(() {
      _isMuted = !_isMuted;
      _remoteRenderer.muted = _isMuted;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        // 상단: 기기 선택 드롭다운
        if (_devices.isNotEmpty)
          Padding(
            padding: const EdgeInsets.only(bottom: 12.0),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                const Text(
                  '시청할 기기 선택',
                  style: TextStyle(fontWeight: FontWeight.bold, fontSize: 14),
                ),
                DropdownButton<String>(
                  value: _selectedDeviceId,
                  items: _devices.map<DropdownMenuItem<String>>((dynamic device) {
                    return DropdownMenuItem<String>(
                      value: device['device_id'],
                      child: Text(device['model'] ?? device['device_id']),
                    );
                  }).toList(),
                  onChanged: _changeDevice,
                ),
              ],
            ),
          ),
          
        // 중앙: 모니터링 영상 (비율 고정)
        AspectRatio(
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
                    RTCVideoView(
                      _remoteRenderer,
                      objectFit: RTCVideoViewObjectFit.RTCVideoViewObjectFitContain,
                      mirror: false,
                    )
                  else if (_isConnecting)
              Center(
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    const CircularProgressIndicator(color: Colors.orange),
                    const SizedBox(height: 16),
                    Text(
                      _statusMsg,
                      style: const TextStyle(color: Colors.white70, fontSize: 13),
                    ),
                  ],
                ),
              )
            else
              Center(
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    const Icon(Icons.videocam_off, color: Colors.white54, size: 40),
                    const SizedBox(height: 12),
                    const Text(
                      '연결된 CCTV가 없습니다.\n공기계에서 캠 모드를 실행해주세요.',
                      textAlign: TextAlign.center,
                      style: TextStyle(color: Colors.white54, fontSize: 13),
                    ),
                    if (_wsError != null) ...[
                      const SizedBox(height: 8),
                      Text(
                        _wsError!,
                        textAlign: TextAlign.center,
                        style: const TextStyle(color: Colors.redAccent, fontSize: 11),
                      ),
                    ],
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
            
            // 음소거 제어 버튼 (연결 완료 시 표시)
            if (_isConnected)
              Positioned(
                bottom: 12,
                right: 12,
                child: GestureDetector(
                  onTap: _toggleMute,
                  child: Container(
                    padding: const EdgeInsets.all(8),
                    decoration: BoxDecoration(
                      color: Colors.black.withOpacity(0.5),
                      shape: BoxShape.circle,
                    ),
                    child: Icon(
                      _isMuted ? Icons.volume_off : Icons.volume_up,
                      color: Colors.white,
                      size: 20,
                    ),
                  ),
                ),
              ),
                ],
              ),
            ),
          ),
        ),
        
        // 하단: 음성 추가 및 송출 버튼
        const SizedBox(height: 16),
        Row(
          children: [
            Expanded(
              child: ElevatedButton.icon(
                onPressed: _isConnected ? _toggleRecording : null,
                icon: Icon(
                  _isRecording ? Icons.stop : Icons.mic,
                  color: Colors.white,
                ),
                label: Text(
                  _isRecording ? '녹음 중지' : '음성 추가하기',
                  style: const TextStyle(fontWeight: FontWeight.bold),
                ),
                style: ElevatedButton.styleFrom(
                  backgroundColor: _isRecording ? Colors.redAccent : Colors.orange,
                  foregroundColor: Colors.white,
                  padding: const EdgeInsets.symmetric(vertical: 12),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(12),
                  ),
                ),
              ),
            ),
            if (_hasVoiceStream) ...[
              const SizedBox(width: 12),
              Expanded(
                child: ElevatedButton.icon(
                  onPressed: _isConnected ? _sendVoice : null,
                  icon: const Icon(Icons.send, color: Colors.white),
                  label: const Text(
                    '아이에게 보내기',
                    style: TextStyle(fontWeight: FontWeight.bold),
                  ),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.blueAccent,
                    foregroundColor: Colors.white,
                    padding: const EdgeInsets.symmetric(vertical: 12),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(12),
                    ),
                  ),
                ),
              ),
              const SizedBox(width: 8),
              IconButton(
                onPressed: _isConnected ? _deleteVoice : null,
                icon: const Icon(Icons.delete_outline, color: Colors.redAccent),
                tooltip: '녹음 삭제',
              ),
            ],
          ],
        ),
      ],
    );
  }
}
