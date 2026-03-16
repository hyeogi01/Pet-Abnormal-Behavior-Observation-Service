import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

class daily_pet extends StatefulWidget {
  final Map<String, dynamic>? petData;
  final String? userId;

  const daily_pet({super.key, required this.petData, this.userId});

  @override
  State<daily_pet> createState() => _DailyPetState();
}

class _DailyPetState extends State<daily_pet> {
  bool _isSaving = false;
  bool _isLoadingStats = true;
  String? _diaryContent;
  String? _errorMessage;
  Map<String, double> _emotionStats = {};
  String _avgMood = "분석 중";
  Color _moodColor = Colors.grey;
  IconData _moodIcon = Icons.sentiment_neutral;

  @override
  void initState() {
    super.initState();
    _fetchDailyStats();
  }

  Future<void> _fetchDailyStats() async {
    final String userId = widget.userId ?? 'test_user';
    final String today = DateTime.now().toIso8601String().substring(0, 10);
    final String baseUrl = 'http://localhost:8080';

    try {
      final response = await http.get(Uri.parse('$baseUrl/api/daily-stats/$userId?date=$today'));
      if (response.statusCode == 200) {
        final result = jsonDecode(response.body);
        if (result['status'] == 'success') {
          final stats = Map<String, double>.from(result['data'] ?? {});
          setState(() {
            _emotionStats = stats;
            _isLoadingStats = false;
            _updateAverageMood();
          });
        }
      }
    } catch (e) {
      print("Error fetching stats: $e");
      setState(() => _isLoadingStats = false);
    }
  }

  void _updateAverageMood() {
    if (_emotionStats.isEmpty) {
      _avgMood = "데이터 없음";
      _moodColor = Colors.grey;
      _moodIcon = Icons.sentiment_neutral;
      return;
    }

    String topEmo = _emotionStats.entries.reduce((a, b) => a.value > b.value ? a : b).key;
    _avgMood = topEmo;
    
    final Map<String, Color> emotionColors = {
      "행복": Colors.green.shade400,
      "활발": Colors.yellow.shade700,
      "불안": Colors.orange.shade400,
      "우울": Colors.purple.shade300,
      "화남": Colors.red.shade400,
      "졸림": Colors.blue.shade300,
      "심심": Colors.grey.shade400,
    };

    final Map<String, IconData> emotionIcons = {
      "행복": Icons.sentiment_very_satisfied,
      "활발": Icons.pets,
      "불안": Icons.sentiment_dissatisfied,
      "우울": Icons.sentiment_very_dissatisfied,
      "화남": Icons.sentiment_very_dissatisfied,
      "졸림": Icons.bedtime,
      "심심": Icons.sentiment_neutral,
    };

    _moodColor = emotionColors[topEmo] ?? Colors.green;
    _moodIcon = emotionIcons[topEmo] ?? Icons.sentiment_satisfied_alt;
  }

  Future<void> _saveDiaryAndGenerate() async {
    final String userId = widget.userId ?? 'test_user';
    final String petType = widget.petData?['pet_type'] ?? 'dog';
    final String baseUrl = 'http://localhost:8080';

    setState(() {
      _isSaving = true;
      _errorMessage = null;
      _diaryContent = null;
    });

    try {
      final response = await http.post(
        Uri.parse('$baseUrl/api/simulate-full-day'),
        body: {
          "user_id": userId,
          "pet_type": petType,
        },
      );
      
      if (response.statusCode != 200) throw Exception('요청 실패');

      final resultJson = jsonDecode(utf8.decode(response.bodyBytes));
      setState(() {
        _diaryContent = resultJson['diary'];
      });

      if (context.mounted) _showDiaryDialog(_diaryContent!);
      _fetchDailyStats();
    } catch (e) {
      setState(() => _errorMessage = e.toString());
    } finally {
      setState(() => _isSaving = false);
    }
  }

  void _showDiaryDialog(String content) {
    showDialog(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('📖 오늘의 일기가 생성됐어요!'),
        content: SingleChildScrollView(child: Text(content)),
        actions: [
          TextButton(onPressed: () => Navigator.pop(ctx), child: const Text('확인'))
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final String petName = widget.petData?['pet_name'] ?? '콩이';
    return Scaffold(
      backgroundColor: const Color(0xFFF9F9F9),
      appBar: AppBar(
        backgroundColor: Colors.blue,
        elevation: 0,
        title: const Text('일상 행동 일기', style: TextStyle(color: Colors.white, fontSize: 16, fontWeight: FontWeight.bold)),
        centerTitle: true,
      ),
      body: SingleChildScrollView(
        child: Padding(
          padding: const EdgeInsets.all(16.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              _buildMoodCard(),
              const SizedBox(height: 24),

              _buildSectionTitle('📸 오늘의 순간들'),
              _buildPhotoGallery(),
              const SizedBox(height: 12),

              _buildEmotionStatistics(petName),
              const SizedBox(height: 24),

              _buildSectionTitle('💬 $petName야 오늘은 어땠어?'),
              _buildAISummaryCard(petName),
              const SizedBox(height: 24),

              _buildTeacherAdviceCard(petName),
              const SizedBox(height: 24),

              _buildSectionTitle('보호자 메모'),
              _buildMemoField(),
              const SizedBox(height: 30),

              _buildBottomButton(context),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildSectionTitle(String title) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 12.0),
      child: Text(title, style: const TextStyle(fontSize: 15, fontWeight: FontWeight.bold, color: Colors.black87)),
    );
  }

  Widget _buildEmotionStatistics(String petName) {
    if (_isLoadingStats) {
      return Container(
        height: 80,
        alignment: Alignment.center,
        child: const CircularProgressIndicator(color: Colors.orange, strokeWidth: 2),
      );
    }

    if (_emotionStats.isEmpty) {
      return const SizedBox.shrink();
    }

    final Map<String, Color> emotionColors = {
      "행복": Colors.green.shade400,
      "활발": Colors.yellow.shade600,
      "불안": Colors.orange.shade400,
      "우울": Colors.purple.shade300,
      "화남": Colors.red.shade400,
      "졸림": Colors.blue.shade300,
      "심심": Colors.grey.shade400,
      "기타": Colors.blueGrey.shade200,
    };

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Text('오늘 하루 $petName의 기분!', style: const TextStyle(fontSize: 14, fontWeight: FontWeight.bold, color: Colors.black87)),
            const Icon(Icons.chevron_right, color: Colors.grey, size: 20),
          ],
        ),
        const SizedBox(height: 12),
        Container(
          padding: const EdgeInsets.all(16.0),
          decoration: BoxDecoration(
            color: Colors.white,
            borderRadius: BorderRadius.circular(15),
            boxShadow: [BoxShadow(color: Colors.black.withOpacity(0.03), blurRadius: 10, offset: const Offset(0, 4))],
          ),
          child: Column(
            children: [
              ClipRRect(
                borderRadius: BorderRadius.circular(10),
                child: SizedBox(
                  height: 14,
                  width: double.infinity,
                  child: Row(
                    children: _emotionStats.entries.map((entry) {
                      return Expanded(
                        flex: (entry.value * 10).toInt().clamp(1, 1000),
                        child: Container(color: emotionColors[entry.key] ?? Colors.grey),
                      );
                    }).toList(),
                  ),
                ),
              ),
              const SizedBox(height: 12),
              Wrap(
                spacing: 12,
                runSpacing: 8,
                children: _emotionStats.entries.map((entry) {
                  return Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Container(
                        width: 10,
                        height: 10,
                        decoration: BoxDecoration(shape: BoxShape.circle, color: emotionColors[entry.key] ?? Colors.grey),
                      ),
                      const SizedBox(width: 4),
                      Text("${entry.key} ${entry.value.toInt()}%", style: TextStyle(fontSize: 11, color: Colors.grey.shade600)),
                    ],
                  );
                }).toList(),
              ),
            ],
          ),
        ),
      ],
    );
  }

  Widget _buildMoodCard() {
    return Container(
      padding: const EdgeInsets.all(18.0),
      decoration: BoxDecoration(
        color: const Color(0xFFFFFDF8),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: Colors.orange.withOpacity(0.1)),
      ),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text('평균 기분', style: TextStyle(color: Colors.grey.shade600, fontSize: 12)),
              const SizedBox(height: 4),
              Text(_avgMood, style: TextStyle(color: _moodColor, fontSize: 22, fontWeight: FontWeight.bold)),
            ],
          ),
          Icon(_moodIcon, color: _moodColor.withOpacity(0.8), size: 40),
        ],
      ),
    );
  }

  Widget _buildPhotoGallery() {
    return Column(
      children: [
        Container(
          height: 175,
          width: double.infinity,
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(15),
            color: Colors.grey[200],
            image: const DecorationImage(image: NetworkImage('https://via.placeholder.com/600x400'), fit: BoxFit.cover),
          ),
        ),
        const SizedBox(height: 12),
        GridView.builder(
          shrinkWrap: true,
          physics: const NeverScrollableScrollPhysics(),
          itemCount: 3,
          gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
            crossAxisCount: 3,
            crossAxisSpacing: 8,
            mainAxisSpacing: 8,
            childAspectRatio: 1,
          ),
          itemBuilder: (context, index) => Container(
            decoration: BoxDecoration(
              borderRadius: BorderRadius.circular(12),
              color: Colors.grey[200],
              image: const DecorationImage(image: NetworkImage('https://via.placeholder.com/150'), fit: BoxFit.cover),
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildAISummaryCard(String petName) {
    return Container(
      padding: const EdgeInsets.all(16.0),
      decoration: BoxDecoration(color: const Color(0xFFF0F5FF), borderRadius: BorderRadius.circular(12)),
      child: Text(
        _diaryContent ?? '오늘 $petName의 활동 내역을 기반으로 일기가 생성됩니다.',
        style: const TextStyle(color: Colors.blueGrey, fontSize: 13, height: 1.5),
      ),
    );
  }

  Widget _buildTeacherAdviceCard(String petName) {
    return Container(
      padding: const EdgeInsets.all(16.0),
      decoration: BoxDecoration(
        gradient: LinearGradient(colors: [Colors.blue.shade400, Colors.indigo.shade400]),
        borderRadius: BorderRadius.circular(15),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              const CircleAvatar(radius: 15, backgroundColor: Colors.white, child: Icon(Icons.analytics_outlined, size: 18, color: Colors.blue)),
              const SizedBox(width: 10),
              Text('$petName의 행동 분석 레포트', style: const TextStyle(color: Colors.white, fontWeight: FontWeight.bold)),
            ],
          ),
          const SizedBox(height: 12),
          Container(
            padding: const EdgeInsets.all(12.0),
            decoration: BoxDecoration(color: Colors.white.withOpacity(0.15), borderRadius: BorderRadius.circular(10)),
            child: Text(
              '분석 결과, 오늘 $petName는 평소보다 $_avgMood한 상태를 많이 보였습니다. 보호자님의 따뜻한 관심이 필요합니다.',
              style: const TextStyle(color: Colors.white, fontSize: 12),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildMemoField() {
    return TextField(
      maxLines: 4,
      decoration: InputDecoration(
        hintText: '특이 사항을 기록하세요...',
        hintStyle: const TextStyle(fontSize: 13, color: Colors.grey),
        filled: true,
        fillColor: Colors.white,
        border: OutlineInputBorder(borderRadius: BorderRadius.circular(12), borderSide: const BorderSide(color: Colors.black12)),
        enabledBorder: OutlineInputBorder(borderRadius: BorderRadius.circular(12), borderSide: const BorderSide(color: Colors.black12)),
      ),
    );
  }

  Widget _buildBottomButton(BuildContext context) {
    return SizedBox(
      width: double.infinity,
      height: 50,
      child: ElevatedButton(
        onPressed: _isSaving ? null : _saveDiaryAndGenerate,
        style: ElevatedButton.styleFrom(
          backgroundColor: Colors.orange,
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
        ),
        child: _isSaving
            ? const SizedBox(width: 20, height: 20, child: CircularProgressIndicator(color: Colors.white, strokeWidth: 2))
            : const Text('기록 저장 및 일기 생성', style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold)),
      ),
    );
  }
}