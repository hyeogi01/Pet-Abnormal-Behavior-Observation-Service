import 'dart:async';

class WebSocketWrapper {
  void send(String data) {}
  Stream<dynamic> get stream => const Stream.empty();
  void close() {}
}

Future<WebSocketWrapper> connectWebSocket(String url) async {
  throw UnsupportedError('WebSocket not supported on this platform');
}
