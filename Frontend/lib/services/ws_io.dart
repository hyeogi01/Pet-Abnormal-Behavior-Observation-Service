import 'dart:async';
import 'dart:io' as io;

class WebSocketWrapper {
  final io.WebSocket _ws;
  final StreamController<dynamic> _controller = StreamController.broadcast();

  WebSocketWrapper._(this._ws) {
    _ws.listen(
      (data) => _controller.add(data),
      onError: (e) => _controller.addError(e),
      onDone: () => _controller.close(),
    );
  }

  void send(String data) => _ws.add(data);
  Stream<dynamic> get stream => _controller.stream;
  void close() => _ws.close();
}

Future<WebSocketWrapper> connectWebSocket(String url) async {
  final ws = await io.WebSocket.connect(
    url,
    headers: {'ngrok-skip-browser-warning': '69420'},
  );
  return WebSocketWrapper._(ws);
}
