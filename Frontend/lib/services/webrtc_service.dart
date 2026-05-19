import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:flutter_webrtc/flutter_webrtc.dart';
import 'package:web_socket_channel/web_socket_channel.dart';

class WebRTCService {
  WebSocketChannel? _channel;
  RTCPeerConnection? _peerConnection;
  
  final String userId;
  final String deviceId;
  final String signalingUrl; // e.g. ws://localhost:8080
  
  // Callbacks
  Function(MediaStream stream)? onRemoteStream;
  Function(RTCIceConnectionState state)? onConnectionState;
  
  // Is this the sender (camera) or receiver (viewer)?
  final bool isSender;

  WebRTCService({
    required this.userId,
    required this.deviceId,
    required this.signalingUrl,
    required this.isSender,
  });

  Future<void> init(MediaStream? localStream) async {
    // 1. WebSocket Connect
    final wsUrl = '$signalingUrl/ws/webrtc/$userId/$deviceId';
    debugPrint('[WebRTC] Connecting to Signaling Server: $wsUrl');
    
    try {
      _channel = WebSocketChannel.connect(Uri.parse(wsUrl));
      debugPrint('[WebRTC] WebSocket connection established.');
    } catch (e) {
      debugPrint('[WebRTC] WebSocket Connection Error: $e');
    }

    // 2. Create PeerConnection
    Map<String, dynamic> configuration = {
      'iceServers': [
        {'urls': 'stun:stun.l.google.com:19302'},
      ]
    };

    _peerConnection = await createPeerConnection(configuration);
    debugPrint('[WebRTC] PeerConnection created.');

    // 3. Setup Callbacks
    _peerConnection?.onIceConnectionState = (state) {
      debugPrint('[WebRTC] IceConnectionState: $state');
      onConnectionState?.call(state);
    };

    _peerConnection?.onIceCandidate = (candidate) {
      debugPrint('[WebRTC] Sending IceCandidate');
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
      if (!isSender && event.streams.isNotEmpty) {
        onRemoteStream?.call(event.streams[0]);
      }
    };

    // 4. Add Local Stream Tracks (if Sender)
    if (isSender && localStream != null) {
      localStream.getTracks().forEach((track) {
        _peerConnection?.addTrack(track, localStream);
      });
    }

    // 5. Listen to WebSocket Signaling Messages
    _channel?.stream.listen((message) {
      _handleSignalingMessage(message);
    });

    // 6. Create Offer (if Sender)
    if (isSender) {
      await _createOffer();
    }
  }

  Future<void> _createOffer() async {
    if (_peerConnection == null) return;
    debugPrint('[WebRTC] Creating Offer...');
    RTCSessionDescription offer = await _peerConnection!.createOffer();
    await _peerConnection!.setLocalDescription(offer);
    
    _sendMessage({
      'type': 'offer',
      'sdp': offer.sdp,
    });
    debugPrint('[WebRTC] Offer sent.');
  }

  Future<void> _handleSignalingMessage(String message) async {
    debugPrint('[WebRTC] Received message: $message');
    try {
      final data = jsonDecode(message);
      final String type = data['type'];

      switch (type) {
        case 'offer':
          if (!isSender && _peerConnection != null) {
            debugPrint('[WebRTC] Received Offer, creating Answer...');
            await _peerConnection!.setRemoteDescription(
              RTCSessionDescription(data['sdp'], 'offer'),
            );
            RTCSessionDescription answer = await _peerConnection!.createAnswer();
            await _peerConnection!.setLocalDescription(answer);
            _sendMessage({
              'type': 'answer',
              'sdp': answer.sdp,
            });
            debugPrint('[WebRTC] Answer sent.');
          }
          break;
        case 'answer':
          if (isSender && _peerConnection != null) {
            debugPrint('[WebRTC] Received Answer, setting remote description.');
            await _peerConnection!.setRemoteDescription(
              RTCSessionDescription(data['sdp'], 'answer'),
            );
          }
          break;
        case 'candidate':
          debugPrint('[WebRTC] Received IceCandidate, adding to connection.');
          final candidateData = data['candidate'];
          RTCIceCandidate candidate = RTCIceCandidate(
            candidateData['candidate'],
            candidateData['sdpMid'],
            candidateData['sdpMLineIndex'],
          );
          await _peerConnection?.addCandidate(candidate);
          break;
        case 'viewer_joined':
          if (isSender) {
            debugPrint('[WebRTC] Viewer joined, re-negotiating (sending new Offer)...');
            await _createOffer();
          }
          break;
      }
    } catch (e) {
      debugPrint('[WebRTC] Signaling Handle Error: $e');
    }
  }

  void _sendMessage(Map<String, dynamic> data) {
    if (_channel != null) {
      final msg = jsonEncode(data);
      _channel!.sink.add(msg);
      debugPrint('[WebRTC] Sent message: ${data['type']}');
    } else {
      debugPrint('[WebRTC] Error: Cannot send message, WebSocket is null.');
    }
  }
  
  // Notify sender that viewer joined so it can send offer again
  void notifyViewerJoined() {
    if (!isSender) {
      _sendMessage({'type': 'viewer_joined'});
    }
  }

  void dispose() {
    _channel?.sink.close();
    _peerConnection?.close();
  }
}
