class Config {
  // ngrok 고정 도메인 적용
  static const String serverUrl = "https://tissue-leverage-diocese.ngrok-free.dev";
  
  // API 엔드포인트 기본 주소
  static String get apiBaseUrl => serverUrl;
  
  // WebRTC 시그널링 주소 (ws/wss)
  static String get signalingUrl {
    if (serverUrl.startsWith('https')) {
      return serverUrl.replaceFirst('https', 'wss');
    } else {
      return serverUrl.replaceFirst('http', 'ws');
    }
  }

  // ngrok 무료 버전 사용 시 브라우저 경고 페이지를 건너뛰기 위한 헤더
  static Map<String, String> get ngrokHeaders => {
    'ngrok-skip-browser-warning': '69420',
    'Content-Type': 'application/json',
  };
}
