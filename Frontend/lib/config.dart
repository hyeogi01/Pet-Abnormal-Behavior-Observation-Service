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

  // ngrok 무료 버전 사용 시 브라우저 경고 페이지를 건너뛰기 위한 헤더 (JSON 요청용)
  static Map<String, String> get ngrokHeaders => {
    'ngrok-skip-browser-warning': '69420',
    'Content-Type': 'application/json',
  };

  // Form 데이터 전송용 헤더 (Content-Type 미포함 → http 패키지가 자동으로 form-urlencoded 설정)
  static Map<String, String> get ngrokFormHeaders => {
    'ngrok-skip-browser-warning': '69420',
  };

  // 백엔드에서 반환된 MinIO(localhost 또는 minio) 주소를 
  // 외부 접속용 serverUrl 주소로 동적 치환해주는 헬퍼 메서드
  static String resolveImageUrl(String? url) {
    if (url == null || url.isEmpty) return '';
    
    if (url.contains('localhost:9000') || url.contains('minio:9000')) {
      final host = serverUrl.replaceAll('https://', '').replaceAll('http://', '').split('/')[0];
      final protocol = serverUrl.startsWith('https') ? 'https' : 'http';
      
      return url
          .replaceFirst(RegExp(r'https?://localhost:9000'), '$protocol://$host')
          .replaceFirst(RegExp(r'https?://minio:9000'), '$protocol://$host');
    }
    return url;
  }
}
