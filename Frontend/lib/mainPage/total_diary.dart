import 'package:flutter/material.dart';
import 'dart:convert';
import 'package:http/http.dart' as http;
import 'diary_detail.dart';
import 'daily_pet.dart';
import '../main.dart'; // petData 등에 접근하기 위해 필요할 수 있으나userId와 initialDate 위주로 처리
class DiaryListPage extends StatefulWidget {
  final String userId;
  const DiaryListPage({super.key, required this.userId});

  @override
  State<DiaryListPage> createState() => _DiaryListPageState();
}

class _DiaryListPageState extends State<DiaryListPage> {
  final String baseUrl = 'http://localhost:8080'; // !IMPORTANT: 안드로이드 실기기 IP 입력 부분 (예: 192.168.0.X:8080)
  List<dynamic> diaries = [];
  bool isLoading = true;

  @override
  void initState() {
    super.initState();
    _fetchTotalDiaries();
  }

  Future<void> _fetchTotalDiaries() async {
    // limit=0 means fetch all
    final url = Uri.parse('$baseUrl/api/daily-diaries/${widget.userId}?limit=0');
    try {
      final response = await http.get(url);
      if (response.statusCode == 200) {
        final decoded = jsonDecode(response.body);
        if (decoded['status'] == 'success') {
          setState(() {
            diaries = decoded['data'] ?? [];
            isLoading = false;
          });
          return;
        }
      }
    } catch (e) {
      print('Diary list fetch error: $e');
    }
    setState(() => isLoading = false);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFFF8F8F8),
      appBar: AppBar(
        backgroundColor: Colors.white,
        elevation: 0,
        leading: IconButton(
          icon: const Icon(Icons.arrow_back_ios, color: Colors.black, size: 20),
          onPressed: () => Navigator.pop(context),
        ),
        title: const Text('전체 일기 목록', style: TextStyle(color: Colors.black, fontSize: 16, fontWeight: FontWeight.bold)),
        centerTitle: true,
      ),
      body: isLoading
          ? const Center(child: CircularProgressIndicator())
          : diaries.isEmpty
              ? const Center(child: Text('작성된 일기가 없습니다.'))
              : ListView.builder(
                  padding: const EdgeInsets.all(16),
                  itemCount: diaries.length,
                  itemBuilder: (context, index) {
                    final diary = diaries[index];
                    return _buildDiaryListItem(
                        context,
                        diary['date'] ?? '알 수 없는 날짜',
                        diary['content'] ?? '내용 없음',
                        diary['hasWarning'] ?? false, // Can be expanded based on log analysis
                        diary
                    );
                  },
                ),
    );
  }

  // 리스트 아이템 위젯 (기존 메인 디자인 유지 및 확장)
  Widget _buildDiaryListItem(BuildContext context, String date, String content, bool hasWarning, Map<String, dynamic> fullData) {
    return GestureDetector(
      onTap: () {
        Navigator.push(
          context,
          MaterialPageRoute(
            builder: (context) => daily_pet(
              petData: null, // 필요 시 상위에서 넘겨받거나 로컬에서 fetch
              userId: widget.userId,
              initialDate: date,
            ),
          ),
        );
      },
      child: Container(
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
            ),
            child: ClipRRect(
              borderRadius: BorderRadius.circular(10),
              child: fullData['video_url'] != null && fullData['video_url'] != ''
                  ? Image.network(
                      fullData['video_url'],
                      fit: BoxFit.cover,
                      errorBuilder: (context, error, stackTrace) =>
                          const Icon(Icons.pets, color: Colors.grey),
                    )
                  : const Icon(Icons.pets, color: Colors.grey),
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
                Text(
                  content,
                  style: const TextStyle(color: Colors.grey, fontSize: 11),
                  maxLines: 2,
                  overflow: TextOverflow.ellipsis,
                ),
                const SizedBox(height: 6),
                Row(
                  children: [
                    const Icon(Icons.pets, size: 14, color: Colors.green),
                    const Text(' AI 일기 기록됨', style: TextStyle(fontSize: 11)),
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
          const Icon(
            Icons.menu_book,
            color: Colors.grey,
            size: 28,
          ),
        ],
      ),
      ),
    );
  }
}