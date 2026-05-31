import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'package:pet_diary/config.dart';
import 'monitoring_view.dart';

class MonitoringPage extends StatefulWidget {
  final String userId;
  final Map<String, dynamic>? petData;

  const MonitoringPage({
    super.key,
    required this.userId,
    this.petData,
  });

  @override
  State<MonitoringPage> createState() => _MonitoringPageState();
}

class _MonitoringPageState extends State<MonitoringPage> {
  List<dynamic> _connectedDevices = [];
  String? _selectedDeviceId;
  bool _isLoadingDevices = true;

  @override
  void initState() {
    super.initState();
    _fetchDevices();
  }

  Future<void> _fetchDevices() async {
    try {
      final url = Uri.parse('${Config.apiBaseUrl}/api/devices/list/${widget.userId}');
      final response = await http.get(url, headers: Config.ngrokHeaders);
      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        if (data['status'] == 'success') {
          final devices = data['data'] as List<dynamic>;
          setState(() {
            _connectedDevices = devices;
            if (devices.isNotEmpty) {
              _selectedDeviceId = devices[0]['device_id'];
            }
            _isLoadingDevices = false;
          });
          return;
        }
      }
    } catch (e) {
      debugPrint('기기 목록 로드 실패: $e');
    }
    setState(() => _isLoadingDevices = false);
  }

  @override
  Widget build(BuildContext context) {
    final petName = widget.petData?['pet_name'] ?? '우리 아이';

    return Scaffold(
      backgroundColor: Colors.white,
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const SizedBox(height: 10),
            Text(
              '$petName 실시간 모니터링',
              style: const TextStyle(
                fontSize: 22,
                fontWeight: FontWeight.bold,
                color: Colors.black87,
              ),
            ),
            const SizedBox(height: 8),
            const Text(
              '연결된 기기에서 전송되는 실시간 영상을 확인하세요.',
              style: TextStyle(fontSize: 14, color: Colors.grey),
            ),
            const SizedBox(height: 20),

            if (_isLoadingDevices)
              const Center(
                child: Padding(
                  padding: EdgeInsets.symmetric(vertical: 40),
                  child: CircularProgressIndicator(color: Colors.orange),
                ),
              )
            else if (_connectedDevices.isEmpty)
              _buildNoDeviceCard()
            else ...[
              if (_connectedDevices.length > 1) ...[
                _buildDeviceSelector(),
                const SizedBox(height: 16),
              ],
              MonitoringView(
                key: ValueKey(_selectedDeviceId),
                userId: widget.userId,
                targetCamDeviceId: _selectedDeviceId!,
              ),
              const SizedBox(height: 40),
              _buildInfoCard(
                title: '기기 상태',
                content: '${_connectedDevices.length}개의 기기가 연결되어 있습니다.',
                icon: Icons.info_outline,
                color: Colors.blueAccent,
              ),
              const SizedBox(height: 16),
              _buildInfoCard(
                title: '네트워크 환경',
                content: '연결 상태가 안정적입니다.',
                icon: Icons.wifi,
                color: Colors.green,
              ),
            ],
          ],
        ),
      ),
    );
  }

  Widget _buildDeviceSelector() {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 4),
      decoration: BoxDecoration(
        border: Border.all(color: Colors.blueAccent.withOpacity(0.3)),
        borderRadius: BorderRadius.circular(12),
        color: Colors.blueAccent.withOpacity(0.05),
      ),
      child: DropdownButton<String>(
        value: _selectedDeviceId,
        isExpanded: true,
        underline: const SizedBox(),
        icon: const Icon(Icons.expand_more, color: Colors.blueAccent),
        items: _connectedDevices.map((device) {
          return DropdownMenuItem<String>(
            value: device['device_id'] as String,
            child: Row(
              children: [
                const Icon(Icons.videocam, color: Colors.blueAccent, size: 18),
                const SizedBox(width: 8),
                Expanded(
                  child: Text(
                    device['model'] ?? '알 수 없는 기기',
                    style: const TextStyle(fontSize: 14),
                    overflow: TextOverflow.ellipsis,
                  ),
                ),
              ],
            ),
          );
        }).toList(),
        onChanged: (deviceId) {
          if (deviceId != null) {
            setState(() => _selectedDeviceId = deviceId);
          }
        },
      ),
    );
  }

  Widget _buildNoDeviceCard() {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.symmetric(vertical: 48, horizontal: 24),
      decoration: BoxDecoration(
        color: Colors.grey[100],
        borderRadius: BorderRadius.circular(15),
        border: Border.all(color: Colors.grey.withOpacity(0.2)),
      ),
      child: const Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(Icons.videocam_off, color: Colors.grey, size: 48),
          SizedBox(height: 16),
          Text(
            '연결된 CCTV가 없습니다.',
            style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold, color: Colors.black54),
          ),
          SizedBox(height: 8),
          Text(
            '마이페이지에서 기기를 추가해주세요.',
            style: TextStyle(fontSize: 13, color: Colors.grey),
          ),
        ],
      ),
    );
  }

  Widget _buildInfoCard({
    required String title,
    required String content,
    required IconData icon,
    required Color color,
  }) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: color.withOpacity(0.05),
        borderRadius: BorderRadius.circular(15),
        border: Border.all(color: color.withOpacity(0.1)),
      ),
      child: Row(
        children: [
          Icon(icon, color: color, size: 24),
          const SizedBox(width: 16),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  title,
                  style: TextStyle(
                    fontWeight: FontWeight.bold,
                    fontSize: 15,
                    color: color,
                  ),
                ),
                const SizedBox(height: 4),
                Text(
                  content,
                  style: const TextStyle(fontSize: 13, color: Colors.black54),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
