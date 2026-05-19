import 'package:flutter/material.dart';
import 'monitoring_view.dart';

class MonitoringPage extends StatelessWidget {
  final String userId;
  final Map<String, dynamic>? petData;

  const MonitoringPage({
    super.key,
    required this.userId,
    this.petData,
  });

  @override
  Widget build(BuildContext context) {
    String petName = petData?['pet_name'] ?? '우리 아이';

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
            const SizedBox(height: 30),
            
            // 모니터링 뷰 위젯
            MonitoringView(userId: userId),
            
            const SizedBox(height: 40),
            
            // 추가 정보 카드 (기능 확장성 고려)
            _buildInfoCard(
              title: '기기 상태',
              content: '현재 1개의 기기가 연결되어 송출 중입니다.',
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
        ),
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
