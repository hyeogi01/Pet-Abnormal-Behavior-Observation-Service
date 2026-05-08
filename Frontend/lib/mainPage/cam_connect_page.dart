import 'package:flutter/material.dart';
import 'package:flutter/foundation.dart';
import 'package:mobile_scanner/mobile_scanner.dart';
import 'package:http/http.dart' as http;
import 'package:device_info_plus/device_info_plus.dart';
import 'dart:convert';
import 'cam_sender_page.dart';
import 'package:pet_diary/config.dart';

/// 공기계(송출 기기)에서 사용하는 페이지
/// QR 코드를 스캔하거나 6자리 코드를 입력하여 메인 기기와 페어링합니다.
class CamConnectPage extends StatefulWidget {
  const CamConnectPage({super.key});

  @override
  State<CamConnectPage> createState() => _CamConnectPageState();
}

class _CamConnectPageState extends State<CamConnectPage> with SingleTickerProviderStateMixin {
  late TabController _tabController;
  final TextEditingController _codeController = TextEditingController();
  bool _isConnecting = false;
  String? _errorMessage;
  // final String baseUrl = "http://localhost:8080"; // Config 사용으로 제거

  @override
  void initState() {
    super.initState();
    _tabController = TabController(length: 2, vsync: this);
  }

  @override
  void dispose() {
    _tabController.dispose();
    _codeController.dispose();
    super.dispose();
  }

  Future<String> _getDeviceModel() async {
    try {
      DeviceInfoPlugin deviceInfo = DeviceInfoPlugin();
      if (kIsWeb) {
        WebBrowserInfo webBrowserInfo = await deviceInfo.webBrowserInfo;
        return '${webBrowserInfo.browserName.name} (Web)';
      } else if (defaultTargetPlatform == TargetPlatform.android) {
        AndroidDeviceInfo androidInfo = await deviceInfo.androidInfo;
        return androidInfo.model;
      } else if (defaultTargetPlatform == TargetPlatform.iOS) {
        IosDeviceInfo iosInfo = await deviceInfo.iosInfo;
        return iosInfo.utsname.machine;
      }
    } catch (e) {
      debugPrint('Error getting device info: $e');
    }
    return 'Unknown Device';
  }

  /// QR 코드에서 읽은 데이터 또는 수동 입력 코드로 페어링 시도
  Future<void> _connectWithCode(String code) async {
    if (code.isEmpty) return;

    setState(() {
      _isConnecting = true;
      _errorMessage = null;
    });

    try {
      String deviceModel = await _getDeviceModel();
      final url = Uri.parse('${Config.apiBaseUrl}/api/devices/pair');
      final response = await http.post(
        url,
        headers: Config.ngrokHeaders,
        body: jsonEncode({
          'pairing_code': code,
          'device_model': deviceModel,
        }),
      ).timeout(const Duration(seconds: 10));

      final result = jsonDecode(response.body);

      if (response.statusCode == 200 && result['status'] == 'success') {
        if (!mounted) return;
        final deviceId = result['device_id'] ?? 'cam_${DateTime.now().millisecondsSinceEpoch}';
        final userId = result['user_id'] ?? '';

        Navigator.pushReplacement(
          context,
          MaterialPageRoute(
            builder: (context) => CamSenderPage(
              deviceId: deviceId,
              userId: userId,
            ),
          ),
        );
      } else {
        setState(() {
          _errorMessage = result['message'] ?? '연결에 실패했습니다. 코드를 확인해 주세요.';
        });
      }
    } catch (e) {
      setState(() {
        _errorMessage = '서버 연결 실패: 백엔드가 켜져있는지 확인해주세요.';
      });
    } finally {
      setState(() => _isConnecting = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF1A1A2E),
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        elevation: 0,
        leading: IconButton(
          icon: const Icon(Icons.arrow_back_ios, color: Colors.white),
          onPressed: () => Navigator.pop(context),
        ),
        title: const Text(
          '캠 연결하기',
          style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold),
        ),
        centerTitle: true,
      ),
      body: Column(
        children: [
          const SizedBox(height: 16),
          // 안내 텍스트
          const Padding(
            padding: EdgeInsets.symmetric(horizontal: 24),
            child: Text(
              '메인 기기에서 생성된 QR 코드를 스캔하거나\n연결 코드를 입력해주세요.',
              textAlign: TextAlign.center,
              style: TextStyle(color: Colors.white70, fontSize: 14, height: 1.5),
            ),
          ),
          const SizedBox(height: 24),

          // 탭 바
          Container(
            margin: const EdgeInsets.symmetric(horizontal: 24),
            decoration: BoxDecoration(
              color: Colors.white.withOpacity(0.1),
              borderRadius: BorderRadius.circular(12),
            ),
            child: TabBar(
              controller: _tabController,
              indicator: BoxDecoration(
                color: Colors.orange,
                borderRadius: BorderRadius.circular(12),
              ),
              indicatorSize: TabBarIndicatorSize.tab,
              labelColor: Colors.white,
              unselectedLabelColor: Colors.white54,
              dividerColor: Colors.transparent,
              tabs: const [
                Tab(text: 'QR 스캔'),
                Tab(text: '코드 입력'),
              ],
            ),
          ),
          const SizedBox(height: 24),

          // 에러 메시지
          if (_errorMessage != null)
            Container(
              margin: const EdgeInsets.symmetric(horizontal: 24),
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: Colors.red.withOpacity(0.15),
                borderRadius: BorderRadius.circular(10),
                border: Border.all(color: Colors.red.withOpacity(0.3)),
              ),
              child: Row(
                children: [
                  const Icon(Icons.error_outline, color: Colors.redAccent, size: 20),
                  const SizedBox(width: 8),
                  Expanded(
                    child: Text(
                      _errorMessage!,
                      style: const TextStyle(color: Colors.redAccent, fontSize: 13),
                    ),
                  ),
                ],
              ),
            ),

          // 탭 콘텐츠
          Expanded(
            child: TabBarView(
              controller: _tabController,
              children: [
                _buildQRScanTab(),
                _buildCodeInputTab(),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildQRScanTab() {
    return Padding(
      padding: const EdgeInsets.all(24),
      child: Column(
        children: [
          Expanded(
            child: ClipRRect(
              borderRadius: BorderRadius.circular(20),
              child: Container(
                decoration: BoxDecoration(
                  border: Border.all(color: Colors.orange.withOpacity(0.5), width: 2),
                  borderRadius: BorderRadius.circular(20),
                ),
                child: MobileScanner(
                  onDetect: (BarcodeCapture capture) {
                    final List<Barcode> barcodes = capture.barcodes;
                    for (final barcode in barcodes) {
                      if (barcode.rawValue != null && !_isConnecting) {
                        _connectWithCode(barcode.rawValue!);
                        break;
                      }
                    }
                  },
                ),
              ),
            ),
          ),
          const SizedBox(height: 16),
          Text(
            _isConnecting ? '연결 중...' : '카메라를 QR 코드에 가까이 대세요',
            style: TextStyle(
              color: _isConnecting ? Colors.orange : Colors.white54,
              fontSize: 14,
            ),
          ),
          if (_isConnecting)
            const Padding(
              padding: EdgeInsets.only(top: 12),
              child: CircularProgressIndicator(color: Colors.orange),
            ),
        ],
      ),
    );
  }

  Widget _buildCodeInputTab() {
    return Padding(
      padding: const EdgeInsets.all(24),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          // 아이콘
          Container(
            width: 80,
            height: 80,
            decoration: BoxDecoration(
              color: Colors.orange.withOpacity(0.15),
              shape: BoxShape.circle,
            ),
            child: const Icon(Icons.link_rounded, color: Colors.orange, size: 40),
          ),
          const SizedBox(height: 24),
          const Text(
            '6자리 연결 코드를 입력해주세요',
            style: TextStyle(color: Colors.white70, fontSize: 15),
          ),
          const SizedBox(height: 24),

          // 코드 입력 필드
          TextField(
            controller: _codeController,
            textAlign: TextAlign.center,
            maxLength: 6,
            keyboardType: TextInputType.number,
            style: const TextStyle(
              color: Colors.white,
              fontSize: 28,
              fontWeight: FontWeight.bold,
              letterSpacing: 12,
            ),
            decoration: InputDecoration(
              counterText: '',
              filled: true,
              fillColor: Colors.white.withOpacity(0.1),
              border: OutlineInputBorder(
                borderRadius: BorderRadius.circular(15),
                borderSide: BorderSide.none,
              ),
              focusedBorder: OutlineInputBorder(
                borderRadius: BorderRadius.circular(15),
                borderSide: const BorderSide(color: Colors.orange, width: 2),
              ),
              contentPadding: const EdgeInsets.symmetric(vertical: 20),
            ),
          ),
          const SizedBox(height: 24),

          // 연결 버튼
          SizedBox(
            width: double.infinity,
            height: 52,
            child: ElevatedButton(
              onPressed: _isConnecting
                  ? null
                  : () => _connectWithCode(_codeController.text.trim()),
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.orange,
                foregroundColor: Colors.white,
                disabledBackgroundColor: Colors.orange.withOpacity(0.5),
                elevation: 0,
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(14),
                ),
              ),
              child: _isConnecting
                  ? const SizedBox(
                      width: 24,
                      height: 24,
                      child: CircularProgressIndicator(color: Colors.white, strokeWidth: 2),
                    )
                  : const Text(
                      '연결하기',
                      style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
                    ),
            ),
          ),
        ],
      ),
    );
  }
}
