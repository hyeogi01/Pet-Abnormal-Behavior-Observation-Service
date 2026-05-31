import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:flutter_webrtc/flutter_webrtc.dart';
import 'package:http/http.dart' as http;
import 'package:path_provider/path_provider.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:pet_diary/config.dart';
import 'package:pet_diary/discription/onboarding_page.dart';
import 'package:pet_diary/services/pet_detector.dart';
import 'package:pet_diary/services/webrtc_service.dart';

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

  // 온디바이스 YOLO 감지 + 주기적 캡처
  final PetDetector _petDetector = PetDetector();
  int _recordingIntervalMinutes = 1;
  String _petType = 'dog';
  bool _isInDetectionLoop = false;
  bool _isCapturing = false;
  Timer? _normalTimer;
  Timer? _retryTimer;
  static const int _retryIntervalMinutes = 1;
  static const int _clipDurationSeconds = 5;

  MediaRecorder? _mediaRecorder;

  @override
  void initState() {
    super.initState();
    _initRenderer();
    _petDetector.load();
    _loadUserSettings();
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

  // ──────────────────────────────────────────────────────────────
  // 설정 로드 + 타이머 시작
  // ──────────────────────────────────────────────────────────────

  Future<void> _loadUserSettings() async {
    try {
      final sRes = await http.get(
        Uri.parse('${Config.apiBaseUrl}/api/settings/${widget.userId}'),
        headers: Config.ngrokHeaders,
      ).timeout(const Duration(seconds: 5));
      if (sRes.statusCode == 200) {
        final d = jsonDecode(sRes.body);
        if (d['status'] == 'success' && d['settings'] != null) {
          _recordingIntervalMinutes = d['settings']['recording_interval'] ?? 60;
        }
      }
      final pRes = await http.get(
        Uri.parse('${Config.apiBaseUrl}/user-pet-info/${widget.userId}'),
        headers: Config.ngrokHeaders,
      ).timeout(const Duration(seconds: 5));
      if (pRes.statusCode == 200) {
        final d = jsonDecode(pRes.body);
        if (d['status'] == 'success' && d['data'] != null) {
          _petType = d['data']['pet_type'] ?? 'dog';
        }
      }
    } catch (e) {
      debugPrint('[CamSender] Settings load error: $e');
    } finally {
      _startNormalTimer();
    }
  }

  void _startNormalTimer() {
    _normalTimer?.cancel();
    _normalTimer = Timer.periodic(
      Duration(minutes: _recordingIntervalMinutes),
      (_) {
        if (!_isInDetectionLoop) _startDetectionLoop();
      },
    );
    debugPrint('[CamSender] Normal timer started: every $_recordingIntervalMinutes min');
  }

  // ──────────────────────────────────────────────────────────────
  // 감지 루프 (pet이 감지될 때까지 2분 간격으로 재시도)
  // ──────────────────────────────────────────────────────────────

  void _startDetectionLoop() {
    _isInDetectionLoop = true;
    _tryDetectAndCapture();
  }

  Future<void> _tryDetectAndCapture() async {
    if (_isCapturing || _localStream == null) {
      _scheduleRetry();
      return;
    }
    _isCapturing = true;

    String? tempFilePath;
    try {
      // 1. YOLO용 프레임 캡처
      final videoTracks = _localStream!.getVideoTracks();
      if (videoTracks.isEmpty) {
        _scheduleRetry();
        return;
      }
      final ByteBuffer buf = await videoTracks.first.captureFrame();
      final Uint8List frameBytes = buf.asUint8List();

      // 2. 온디바이스 YOLO 추론
      final petFound = await _petDetector.detect(frameBytes, petType: _petType);
      if (!petFound) {
        debugPrint('[CamSender] No pet → retry in $_retryIntervalMinutes min');
        _scheduleRetry();
        return;
      }

      // 3. 반려동물 감지됨 → 5초 클립 녹화 (영상+오디오)
      debugPrint('[CamSender] Pet detected → recording ${_clipDurationSeconds}s clip');
      final tempDir = await getTemporaryDirectory();
      tempFilePath = '${tempDir.path}/clip_${DateTime.now().millisecondsSinceEpoch}.mp4';

      _mediaRecorder = MediaRecorder();
      await _mediaRecorder!.start(
        tempFilePath,
        videoTrack: videoTracks.first,
        audioChannel: RecorderAudioChannel.INPUT,
      );

      await Future.delayed(const Duration(seconds: _clipDurationSeconds));
      await _mediaRecorder!.stop();
      _mediaRecorder = null;

      // 4. 클립 파일 읽기
      final clipFile = File(tempFilePath);
      final clipBytes = await clipFile.readAsBytes();
      await clipFile.delete();
      tempFilePath = null;

      // 5. 서버로 전송 (PNG → 시각 분석, MP4 → 오디오 분석)
      final req = http.MultipartRequest(
        'POST',
        Uri.parse('${Config.apiBaseUrl}/api/daily-behavior'),
      );
      req.headers.addAll(Config.ngrokFormHeaders);
      req.fields['user_id'] = widget.userId;
      req.fields['pet_type'] = _petType;
      req.fields['timestamp'] = DateTime.now().toIso8601String();
      req.files.add(http.MultipartFile.fromBytes(
        'file',
        frameBytes,
        filename: 'frame_${DateTime.now().millisecondsSinceEpoch}.png',
      ));
      req.files.add(http.MultipartFile.fromBytes(
        'audio_clip',
        clipBytes,
        filename: 'clip_${DateTime.now().millisecondsSinceEpoch}.mp4',
      ));

      final res = await http.Response.fromStream(await req.send());
      if (res.statusCode == 200) {
        final data = jsonDecode(res.body);
        debugPrint('[CamSender] Analysis result: ${data['status']}');
      }

      // 감지 루프 종료 → IDLE 복귀
      _isInDetectionLoop = false;
      _retryTimer?.cancel();

    } catch (e) {
      debugPrint('[CamSender] Capture/analyze error: $e');
      // 녹화 중 에러 시 recorder 정리
      if (_mediaRecorder != null) {
        try { await _mediaRecorder!.stop(); } catch (_) {}
        _mediaRecorder = null;
      }
      // 임시 파일 정리
      if (tempFilePath != null) {
        try { await File(tempFilePath).delete(); } catch (_) {}
      }
      _scheduleRetry();
    } finally {
      _isCapturing = false;
    }
  }

  void _scheduleRetry() {
    _retryTimer?.cancel();
    _retryTimer = Timer(
      const Duration(minutes: _retryIntervalMinutes),
      _tryDetectAndCapture,
    );
  }

  // ──────────────────────────────────────────────────────────────

  void _showExitDialog() {
    showDialog(
      context: context,
      builder: (ctx) => AlertDialog(
        backgroundColor: const Color(0xFF2C2C2E),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
        title: const Text('CCTV 모드 종료', style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold)),
        content: const Text(
          '어떻게 하시겠습니까?',
          style: TextStyle(color: Colors.white70),
        ),
        actions: [
          // 일시 종료 (자격증명 유지 → 다음 실행 시 자동 재연결)
          TextButton(
            onPressed: () {
              Navigator.pop(ctx);
              Navigator.pop(context);
            },
            child: const Text('일시 종료', style: TextStyle(color: Colors.white54)),
          ),
          // 연결 해제 (자격증명 삭제 → 다음 실행 시 QR 재등록 필요)
          TextButton(
            onPressed: () async {
              Navigator.pop(ctx);
              final prefs = await SharedPreferences.getInstance();
              await prefs.remove('cam_device_id');
              await prefs.remove('cam_user_id');
              await prefs.remove('cam_device_model');
              if (!mounted) return;
              Navigator.pushAndRemoveUntil(
                context,
                MaterialPageRoute(builder: (_) => const OnboardingPage()),
                (route) => false,
              );
            },
            child: const Text('연결 해제', style: TextStyle(color: Colors.redAccent, fontWeight: FontWeight.bold)),
          ),
        ],
      ),
    );
  }

  @override
  void dispose() {
    _normalTimer?.cancel();
    _retryTimer?.cancel();
    if (_mediaRecorder != null) {
      _mediaRecorder!.stop().catchError((_) {});
      _mediaRecorder = null;
    }
    _petDetector.dispose();
    _webrtcService?.dispose();
    _localStream?.dispose();
    _localRenderer.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      body: SizedBox.expand(
        child: Stack(
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
                      onPressed: _showExitDialog,
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
      ),
    );
  }
}
