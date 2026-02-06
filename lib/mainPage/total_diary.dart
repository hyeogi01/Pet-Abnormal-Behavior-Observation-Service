import 'package:flutter/material.dart';

class DiaryListPage extends StatelessWidget {
  const DiaryListPage({super.key});

  @override
  Widget build(BuildContext context) {
    // 임시 데이터 리스트 (실제로는 서버나 DB에서 가져오게 됩니다)
    final List<Map<String, dynamic>> diaries = [
      {'date': '2026년 2월 6일', 'day': '목요일', 'activity': 85, 'hasWarning': true},
      {'date': '2026년 2월 5일', 'day': '수요일', 'activity': 72, 'hasWarning': false},
      {'date': '2026년 2월 4일', 'day': '화요일', 'activity': 90, 'hasWarning': false},
      {'date': '2026년 2월 3일', 'day': '월요일', 'activity': 65, 'hasWarning': true},
      {'date': '2026년 2월 2일', 'day': '일요일', 'activity': 40, 'hasWarning': false},
      {'date': '2026년 2월 1일', 'day': '토요일', 'activity': 95, 'hasWarning': false},
      {'date': '2026년 1월 31일', 'day': '금요일', 'activity': 80, 'hasWarning': false},
      {'date': '2026년 1월 30일', 'day': '목요일', 'activity': 70, 'hasWarning': false},
      {'date': '2026년 1월 29일', 'day': '수요일', 'activity': 88, 'hasWarning': true},
      {'date': '2026년 1월 28일', 'day': '화요일', 'activity': 50, 'hasWarning': false},
    ];

    return Scaffold(
      backgroundColor: const Color(0xFFF8F8F8),
      appBar: AppBar(
        backgroundColor: Colors.white,
        elevation: 0,
        leading: IconButton(
          icon: const Icon(Icons.arrow_back_ios, color: Colors.black, size: 20),
          onPressed: () => Navigator.pop(context),
        ),
        title: const Text('전체 일기 목록',
            style: TextStyle(color: Colors.black, fontSize: 16, fontWeight: FontWeight.bold)),
        centerTitle: true,
      ),
      body: ListView.builder(
        padding: const EdgeInsets.all(16),
        itemCount: diaries.length,
        itemBuilder: (context, index) {
          final diary = diaries[index];
          return _buildDiaryListItem(
              diary['date'],
              diary['day'],
              diary['activity'],
              diary['hasWarning']
          );
        },
      ),
    );
  }

  // 리스트 아이템 위젯 (기존 메인 디자인 유지 및 확장)
  Widget _buildDiaryListItem(String date, String day, int activity, bool hasWarning) {
    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(15),
        boxShadow: const [
          BoxShadow(color: Colors.black12, blurRadius: 4, offset: Offset(0, 2))
        ],
      ),
      child: Row(
        children: [
          // 왼쪽 이미지 공간
          Container(
            width: 60,
            height: 60,
            decoration: BoxDecoration(
              color: Colors.grey[200],
              borderRadius: BorderRadius.circular(10),
              image: const DecorationImage(
                image: NetworkImage('https://via.placeholder.com/60'), // 실제 이미지로 교체 가능
                fit: BoxFit.cover,
              ),
            ),
          ),
          const SizedBox(width: 16),
          // 중앙 날짜 및 정보
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(date, style: const TextStyle(fontSize: 15, fontWeight: FontWeight.bold)),
                const SizedBox(height: 2),
                Text(day, style: const TextStyle(color: Colors.grey, fontSize: 12)),
                const SizedBox(height: 6),
                Row(
                  children: [
                    const Icon(Icons.trending_up, size: 14, color: Colors.green),
                    Text(' 활동 $activity', style: const TextStyle(fontSize: 12)),
                    if (hasWarning) ...[
                      const SizedBox(width: 8),
                      Container(
                        padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                        decoration: BoxDecoration(
                          color: Colors.orange[50],
                          borderRadius: BorderRadius.circular(4),
                          border: Border.all(color: Colors.orange[100]!),
                        ),
                        child: const Text('주의사항',
                            style: TextStyle(color: Colors.orange, fontSize: 10, fontWeight: FontWeight.bold)),
                      )
                    ]
                  ],
                )
              ],
            ),
          ),
          // 오른쪽 감정 이모지 아이콘
          Icon(
            activity > 80 ? Icons.sentiment_satisfied_alt : Icons.sentiment_neutral,
            color: Colors.green,
            size: 28,
          ),
        ],
      ),
    );
  }
}