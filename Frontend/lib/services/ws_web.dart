import 'dart:async';
import 'dart:html' as html;

class WebSocketWrapper {
  final html.WebSocket _ws;
  final StreamController<dynamic> _controller = StreamController.broadcast();

  WebSocketWrapper._(this._ws) {
    _ws.onMessage.listen((e) => _controller.add(e.data));
    _ws.onClose.listen((e) {
      if (!_controller.isClosed) _controller.close();
    });
    _ws.onError.listen((e) {
      if (!_controller.isClosed) _controller.addError(Exception('WebSocket 오류'));
    });
  }

  void send(String data) => _ws.send(data);
  Stream<dynamic> get stream => _controller.stream;
  void close() => _ws.close();
}

Future<WebSocketWrapper> connectWebSocket(String url) {
  final completer = Completer<WebSocketWrapper>();
  final ws = html.WebSocket(url);

  late StreamSubscription openSub;
  late StreamSubscription errorSub;

  openSub = ws.onOpen.listen((_) {
    openSub.cancel();
    errorSub.cancel();
    completer.complete(WebSocketWrapper._(ws));
  });

  errorSub = ws.onError.listen((_) {
    openSub.cancel();
    errorSub.cancel();
    if (!completer.isCompleted) {
      completer.completeError(Exception('WebSocket 연결 실패'));
    }
  });

  return completer.future;
}
