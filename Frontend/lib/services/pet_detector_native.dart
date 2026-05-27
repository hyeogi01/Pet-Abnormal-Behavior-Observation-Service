import 'dart:typed_data';
import 'package:flutter/foundation.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;

class PetDetector {
  Interpreter? _interpreter;

  static const int _inputSize = 320;
  static const double _threshold = 0.35;

  // YOLOv8n TFLite output: [1, 84, 2100]
  // 84 = 4(bbox: cx,cy,w,h) + 80(COCO classes)
  // COCO class 15=cat → output row 4+15=19
  // COCO class 16=dog → output row 4+16=20
  static const List<int> _petRows = [19, 20];

  // output[0].length = 84 (first dim after batch), output[0][0].length = 2100
  // 2100 = 40×40 + 20×20 + 10×10 anchor points for 320px input
  static const int _numPredictions = 2100;

  bool get isLoaded => _interpreter != null;

  Future<void> load() async {
    try {
      _interpreter = await Interpreter.fromAsset('assets/models/yolov8n.tflite');
      // 첫 로드 시 출력 형태 확인용 로그 (확인 후 제거 가능)
      debugPrint('[PetDetector] Loaded. Output shape: ${_interpreter!.getOutputTensor(0).shape}');
    } catch (e) {
      // fail-open: 모델 파일 없어도 detect()가 true 반환하여 서비스 유지
      debugPrint('[PetDetector] Load failed (fail-open): $e');
    }
  }

  /// 이미지에서 반려동물(cat/dog) 존재 여부를 반환한다.
  /// 모델 미로드 또는 추론 오류 시 true를 반환하여 데이터 손실을 방지한다.
  Future<bool> detect(Uint8List imageBytes) async {
    if (_interpreter == null) return true;

    try {
      final decoded = img.decodeImage(imageBytes);
      if (decoded == null) return true;

      final resized = img.copyResize(decoded, width: _inputSize, height: _inputSize);

      // Input: [1, 320, 320, 3] float32 (RGB, 0.0~1.0)
      final input = List.generate(1, (_) =>
        List.generate(_inputSize, (y) =>
          List.generate(_inputSize, (x) {
            final p = resized.getPixel(x, y);
            return [p.r / 255.0, p.g / 255.0, p.b / 255.0];
          })
        )
      );

      // Output: [1, 84, 2100]
      final output = [
        List.generate(84, (_) => List<double>.filled(_numPredictions, 0.0))
      ];

      _interpreter!.run(input, output);

      // cat(row 19) 또는 dog(row 20) confidence가 threshold 이상인 예측이 하나라도 있으면 감지
      // NMS 불필요 — 위치가 아닌 존재 여부만 판단하므로 단순 스캔으로 충분
      for (int i = 0; i < _numPredictions; i++) {
        for (final row in _petRows) {
          if (output[0][row][i] >= _threshold) {
            debugPrint('[PetDetector] Pet detected at pred=$i, row=$row, conf=${output[0][row][i].toStringAsFixed(3)}');
            return true;
          }
        }
      }

      debugPrint('[PetDetector] No pet detected in frame.');
      return false;

    } catch (e) {
      debugPrint('[PetDetector] Inference error (fail-open): $e');
      return true;
    }
  }

  void dispose() {
    _interpreter?.close();
    _interpreter = null;
  }
}
