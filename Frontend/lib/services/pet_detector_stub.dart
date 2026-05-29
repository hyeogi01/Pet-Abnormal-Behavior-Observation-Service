import 'dart:typed_data';
import 'package:flutter/foundation.dart';

// Web/desktop stub — tflite_flutter는 dart:ffi 기반으로 웹 미지원
// fail-open 정책 유지: detect()는 항상 true 반환
class PetDetector {
  bool get isLoaded => false;

  Future<void> load() async {
    debugPrint('[PetDetector] Stub mode (web/desktop) — TFLite unavailable');
  }

  Future<bool> detect(Uint8List imageBytes) async => true;

  void dispose() {}
}
