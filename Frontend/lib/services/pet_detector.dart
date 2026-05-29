// dart.library.io → Android/iOS/desktop (네이티브 TFLite 사용)
// 미지원 시(web) → stub으로 폴백 (항상 true 반환)
export 'pet_detector_stub.dart'
    if (dart.library.io) 'pet_detector_native.dart';
