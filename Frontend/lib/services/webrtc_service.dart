import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:flutter_webrtc/flutter_webrtc.dart';
import 'ws_stub.dart'
    if (dart.library.html) 'ws_web.dart'
    if (dart.library.io) 'ws_io.dart';

class WebRTCService {
  WebSocketWrapper? _ws;
  RTCPeerConnection? _peerConnection;
  final List<RTCIceCandidate> _pendingCandidates = [];
  bool _remoteDescriptionSet = false;
  
  final String userId;
  final String deviceId;
  final String signalingUrl;
  
  Function(MediaStream stream)? onRemoteStream;
  Function(RTCIceConnectionState state)? onConnectionState;
  Function(String error)? onError;
  Function(String status)? onStatus;
  
  final bool isSender;
  String? targetDeviceId;

  WebRTCService({
    required this.userId,
    required this.deviceId,
    required this.signalingUrl,
    required this.isSender,
    this.targetDeviceId,
  });

  Future<void> init(MediaStream? localStream) async {
    // 1. WebSocket Connect
    final wsUrl = '$signalingUrl/ws/webrtc/$userId/$deviceId?ngrok-skip-browser-warning=true';
    debugPrint('[WebRTC] Connecting: $wsUrl');
    
    try {
      _ws = await connectWebSocket(wsUrl);
      debugPrint('[WebRTC] WebSocket connected.');
    } catch (e) {
      debugPrint('[WebRTC] WebSocket Error: $e');
      onError?.call('WebSocket 연결 실패: $e');
      return;
    }
    
    // 2. PeerConnection Config
    // TURN 서버 없이는 모바일(LTE/5G) ↔ 웹 간 대칭 NAT 통과 불가
    Map<String, dynamic> configuration = {
      'iceServers': [
        {'urls': 'stun:stun.l.google.com:19302'},
        {'urls': 'stun:stun1.l.google.com:19302'},
        {
          'urls': [
            'turn:openrelay.metered.ca:80',
            'turn:openrelay.metered.ca:443',
            'turn:openrelay.metered.ca:443?transport=tcp',
          ],
          'username': 'openrelayproject',
          'credential': 'openrelayproject',
        },
      ],
      'sdpSemantics': 'unified-plan',
    };

    _peerConnection = await createPeerConnection(configuration);
    debugPrint('[WebRTC] PeerConnection created.');

    // 3. Callbacks
    _peerConnection?.onIceConnectionState = (state) {
      debugPrint('[WebRTC] State: $state');
      onConnectionState?.call(state);
    };

    _peerConnection?.onIceCandidate = (candidate) {
      _sendMessage({
        'type': 'candidate',
        'candidate': {
          'candidate': candidate.candidate,
          'sdpMid': candidate.sdpMid,
          'sdpMLineIndex': candidate.sdpMLineIndex,
        }
      });
    };

    _peerConnection?.onTrack = (event) {
      if (event.streams.isNotEmpty) {
        debugPrint('[WebRTC] OnTrack: ${event.track.kind} (isSender: $isSender)');
        onRemoteStream?.call(event.streams[0]);
      }
    };

    // 4. Setup Directions & Tracks
    if (isSender) {
      // Cam (Pet): Sends Video and Audio
      if (localStream != null) {
        for (var track in localStream.getTracks()) {
          await _peerConnection!.addTrack(track, localStream);
        }
      }
      // Force transceivers to be SendOnly/SendRecv to ensure Pet voice is sent
      var transceivers = await _peerConnection!.getTransceivers();
      for (var t in transceivers) {
        if (t.sender.track?.kind == 'video') {
          await t.setDirection(TransceiverDirection.SendOnly);
        } else if (t.sender.track?.kind == 'audio') {
          await t.setDirection(TransceiverDirection.SendRecv);
        }
      }
    } else {
      // Viewer: 캠의 offer와 m-line이 일치하도록 RecvOnly 트랜시버를 미리 생성
      await _peerConnection!.addTransceiver(
        kind: RTCRtpMediaType.RTCRtpMediaTypeVideo,
        init: RTCRtpTransceiverInit(direction: TransceiverDirection.RecvOnly),
      );
      await _peerConnection!.addTransceiver(
        kind: RTCRtpMediaType.RTCRtpMediaTypeAudio,
        init: RTCRtpTransceiverInit(direction: TransceiverDirection.RecvOnly),
      );
    }

    // 5. Listen
    _ws!.stream.listen(
      (message) => _handleSignalingMessage(message.toString()),
      onError: (e) {
        debugPrint('[WebRTC] WebSocket stream error: $e');
        onError?.call('WebSocket 오류: $e');
      },
      onDone: () {
        debugPrint('[WebRTC] WebSocket closed');
        onError?.call('서버 연결 끊어짐 — 앱 재시작 필요');
      },
    );

    // 6. Initiation
    if (!isSender) {
      onStatus?.call('서버 연결됨. 캠 신호 대기 중...');
      debugPrint('[WebRTC] Viewer sending viewer_joined...');
      _sendMessage({'type': 'viewer_joined'});
    } else {
      debugPrint('[WebRTC] Sender broadcasting cam_ready...');
      _sendMessage({'type': 'cam_ready'});
    }
  }

  Future<void> addAudioTrack(MediaStream audioStream) async {
    if (_peerConnection == null) return;

    debugPrint('[WebRTC] Owner adding voice track...');
    for (var track in audioStream.getAudioTracks()) {
      await _peerConnection!.addTrack(track, audioStream);
    }

    // addTrack 후 생성/연결된 오디오 트랜시버를 찾아 SendRecv로 변경
    var transceivers = await _peerConnection!.getTransceivers();
    for (var t in transceivers) {
      if (t.sender.track?.kind == 'audio') {
        await t.setDirection(TransceiverDirection.SendRecv);
        break;
      }
    }

    await _createOffer();
  }

  Future<void> _createOffer() async {
    if (_peerConnection == null) return;
    if (_peerConnection!.signalingState != RTCSignalingState.RTCSignalingStateStable) {
      debugPrint('[WebRTC] Skip Offer: signalingState=${_peerConnection!.signalingState}');
      return;
    }

    debugPrint('[WebRTC] Creating Offer...');
    RTCSessionDescription offer = await _peerConnection!.createOffer();
    await _peerConnection!.setLocalDescription(offer);
    _sendMessage({'type': 'offer', 'sdp': offer.sdp});
  }

  Future<void> _handleSignalingMessage(String message) async {
    try {
      final data = jsonDecode(message);
      final String type = data['type'];

      switch (type) {
        case 'offer':
          if (_peerConnection != null) {
            debugPrint('[WebRTC] Recv Offer, sending Answer...');
            onStatus?.call('캠 연결 요청 수신. 연결 협상 중...');
            await _peerConnection!.setRemoteDescription(RTCSessionDescription(data['sdp'], 'offer'));
            _remoteDescriptionSet = true;
            await _flushPendingCandidates();
            RTCSessionDescription answer = await _peerConnection!.createAnswer();
            await _peerConnection!.setLocalDescription(answer);
            _sendMessage({'type': 'answer', 'sdp': answer.sdp});
            onStatus?.call('연결 협상 완료. ICE 연결 중...');
          }
          break;
        case 'answer':
          if (_peerConnection != null && _peerConnection!.signalingState == RTCSignalingState.RTCSignalingStateHaveLocalOffer) {
            debugPrint('[WebRTC] Recv Answer, setting remote.');
            await _peerConnection!.setRemoteDescription(RTCSessionDescription(data['sdp'], 'answer'));
            _remoteDescriptionSet = true;
            await _flushPendingCandidates();
          }
          break;
        case 'candidate':
          final candidateData = data['candidate'];
          final candidate = RTCIceCandidate(
            candidateData['candidate'],
            candidateData['sdpMid'],
            candidateData['sdpMLineIndex'], ##
          );
          if (_remoteDescriptionSet) {
            await _peerConnection?.addCandidate(candidate);
          } else {
            _pendingCandidates.add(candidate);
          }
          break;
        case 'viewer_joined':
          if (isSender) {
            debugPrint('[WebRTC] Viewer joined, starting negotiation...');
            await Future.delayed(const Duration(milliseconds: 500));
            await _createOffer();
          }
          break;
        case 'cam_ready':
          if (!isSender) {
            debugPrint('[WebRTC] Cam is ready, re-sending viewer_joined...');
            _sendMessage({'type': 'viewer_joined'});
          }
          break;
      }
    } catch (e) {
      debugPrint('[WebRTC] Signaling Error: $e');
    }
  }

  Future<void> _flushPendingCandidates() async {
    for (final c in _pendingCandidates) {
      await _peerConnection?.addCandidate(c);
    }
    _pendingCandidates.clear();
  }

  void _sendMessage(Map<String, dynamic> data) {
    if (_ws != null) {
      if (targetDeviceId != null && !isSender) data['target'] = targetDeviceId;
      data['sender'] = deviceId;
      _ws!.send(jsonEncode(data));
    }
  }

  void dispose() {
    _ws?.close();
    _peerConnection?.close();
  }
}
