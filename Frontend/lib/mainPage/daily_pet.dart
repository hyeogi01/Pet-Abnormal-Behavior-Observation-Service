import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

class daily_pet extends StatefulWidget {
  final Map<String, dynamic>? petData;
  final String? userId;
  final String? initialDate; // 특정 날짜 (YYYY-MM-DD), 없으면 오늘

  const daily_pet({super.key, required this.petData, this.userId, this.initialDate});

  @override
  State<daily_pet> createState() => _DailyPetState();
}

class _DailyPetState extends State<daily_pet> {
  bool _isSaving = false;
  bool _isLoadingStats = true;
  bool _isLoadingDiary = false;
  String? _diaryContent;
  String? _reportContent;
  String? _memoContent;
  List<String> _imageUrls = [];
  final TextEditingController _memoController = TextEditingController();
  String? _errorMessage;
  Map<String, double> _emotionStats = {};
  String _avgMood = "분석 중";
  Color _moodColor = Colors.grey;
  IconData _moodIcon = Icons.sentiment_neutral;
  late String _currentDate;

  @override
  void initState() {
    super.initState();
    _currentDate = widget.initialDate ?? DateTime.now().toIso8601String().substring(0, 10);
    _fetchData();
  }

  Future<void> _fetchData() async {
    setState(() {
      _isLoadingStats = true;
      _isLoadingDiary = true;
    });
    await Future.wait([
      _fetchDailyStats(),
      _fetchSavedDiary(),
    ]);
    setState(() {
      _isLoadingStats = false;
      _isLoadingDiary = false;
    });
  }

  Future<void> _fetchDailyStats() async {
    final String userId = widget.userId ?? 'test_user';
    final String baseUrl = 'http://localhost:8080';

    debugPrint("Fetching daily stats for userId: $userId, date: $_currentDate");

    try {
      final response = await http.get(Uri.parse('$baseUrl/api/daily-stats/$userId?date=$_currentDate'));
      if (response.statusCode == 200) {
        final result = jsonDecode(response.body);
        if (result['status'] == 'success') {
          final rawData = result['data'] as Map<String, dynamic>? ?? {};
          
          // JSON에서 int가 넘어올 수 있으므로 num으로 받은 뒤 double로 변환
          final stats = rawData.map((key, value) => MapEntry(key, (value as num).toDouble()));
          
          debugPrint("Parsed Emotion Stats: $stats");

          setState(() {
            _emotionStats = stats;
            _updateAverageMood();
          });
        }
      } else {
        debugPrint("Failed to fetch stats: ${response.statusCode}");
      }
    } catch (e) {
      debugPrint("Error fetching stats: $e");
    }
  }

  Future<void> _fetchSavedDiary() async {
    final String userId = widget.userId ?? 'test_user';
    final String baseUrl = 'http://localhost:8080';

    debugPrint("Fetching diary for userId: $userId, date: $_currentDate");

    try {
      final response = await http.get(Uri.parse('$baseUrl/api/daily-diaries/$userId?limit=0'));
      if (response.statusCode == 200) {
        final result = jsonDecode(response.body);
        if (result['status'] == 'success') {
          final List diaries = result['data'] ?? [];
          debugPrint("Total diaries found: ${diaries.length}");
          
          final diary = diaries.firstWhere((d) => d['date'] == _currentDate, orElse: () => null);
          if (diary != null) {
            final String content = diary['pet_diary'] ?? "";
            debugPrint("Found diary for $_currentDate: ${content.length > 20 ? content.substring(0, 20) : content}...");
            setState(() {
              _diaryContent = diary['pet_diary'];
              _reportContent = diary['report'];
              _memoContent = diary['memo']; // This will update the display card
              _memoController.text = _memoContent ?? "";
              _imageUrls = List<String>.from(diary['image_urls'] ?? []);
            });
          } else {
            debugPrint("No diary found for $_currentDate");
            setState(() {
              _diaryContent = null;
              _reportContent = null;
              _memoContent = null;
              _memoController.text = "";
              _imageUrls = [];
            });
          }
        }
      } else {
        debugPrint("Failed to fetch diaries: ${response.statusCode}");
      }
    } catch (e) {
      debugPrint("Error fetching diary: $e");
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
      "기타": Colors.blueGrey.shade300,
    };

    final Map<String, IconData> emotionIcons = {
      "행복": Icons.sentiment_very_satisfied,
      "활발": Icons.pets_outlined,
      "불안": Icons.sentiment_dissatisfied,
      "우울": Icons.sentiment_very_dissatisfied,
      "화남": Icons.error_outline,
      "졸림": Icons.bedtime_outlined,
      "심심": Icons.sentiment_neutral,
      "기타": Icons.help_outline,
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
    });

    try {
      // 1. 시뮬레이션 및 일기 생성 요청 (Form 데이터 방식)
      final response = await http.post(
        Uri.parse('$baseUrl/api/simulate-full-day'),
        body: {
          "user_id": userId,
          "pet_type": petType,
        },
      );
      
      if (response.statusCode != 200) throw Exception('일기 생성 요청 실패: ${response.statusCode}');

      // 2. 생성 완료 후 메모가 있다면 메모도 저장
      if (_memoController.text.isNotEmpty) {
        await _saveMemo(silent: true);
      }

      // 3. 최신 데이터 다시 불러오기
      await _fetchSavedDiary();

      if (context.mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('오늘의 일기가 성공적으로 생성되었습니다!')),
        );
      }
    } catch (e) {
      setState(() => _errorMessage = e.toString());
      if (context.mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('오류 발생: $e')),
        );
      }
    } finally {
      setState(() => _isSaving = false);
    }
  }

  Future<void> _saveMemo({bool silent = false}) async {
    final String userId = widget.userId ?? 'test_user';
    final String baseUrl = 'http://localhost:8080';

    if (!silent) setState(() => _isSaving = true);

    try {
      // JSON 방식이 안 될 경우를 대비해 인코딩 명시
      final response = await http.post(
        Uri.parse('$baseUrl/api/save-memo'),
        headers: {"Content-Type": "application/json; charset=UTF-8"},
        body: jsonEncode({
          "user_id": userId,
          "date": _currentDate,
          "memo": _memoController.text,
        }),
      );
      
      if (response.statusCode == 200) {
        debugPrint("Memo saved successfully");
        setState(() {
          _memoContent = _memoController.text; // Update display card
        });
        if (!silent && context.mounted) {
           ScaffoldMessenger.of(context).showSnackBar(
             const SnackBar(content: Text('메모가 저장되었습니다.')),
           );
        }
      } else {
        throw Exception("Server returned ${response.statusCode}");
      }
    } catch (e) {
      debugPrint("Error saving memo: $e");
      if (!silent && context.mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('메모 저장 실패: $e')),
        );
      }
    } finally {
      if (!silent) setState(() => _isSaving = false);
    }
  }

  void _showDiaryDialog(String content) {
    showDialog(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('📖 오늘의 일기가 생성됐어요!'),
        content: SingleChildScrollView(child: Text(content)),
        actions: [
          TextButton(
            onPressed: () {
              Navigator.pop(ctx);
              Navigator.pop(context, true); 
            },
            child: const Text('확인'),
          )
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final String petName = widget.petData?['pet_name'] ?? '콩이';
    final bool isToday = _currentDate == DateTime.now().toIso8601String().substring(0, 10);

    return Scaffold(
      backgroundColor: const Color(0xFFF9F9F9),
      appBar: AppBar(
        backgroundColor: Colors.blue,
        elevation: 0,
        leading: IconButton(
          icon: const Icon(Icons.arrow_back_ios, color: Colors.white, size: 20),
          onPressed: () => Navigator.pop(context),
        ),
        title: Text(' AI 행동 관찰 일기 ($_currentDate)', style: const TextStyle(color: Colors.white, fontSize: 16, fontWeight: FontWeight.bold)),
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

              // 📝 저장된 보호자 메모 패널 (추가)
              if (_diaryContent != null) ...[
                _buildSectionTitle('📝 저장된 보호자 메모'),
                _buildSavedMemoCard(),
                const SizedBox(height: 24),
              ],

              // 오늘일 때만 메모 입력란과 버튼 표시
              if (isToday) ...[
                _buildSectionTitle('✍️ 보호자 메모 작성'),
                _buildMemoField(),
                const SizedBox(height: 30),
                _buildBottomButton(context),
              ] else ...[
                 Center(
                   child: Text(
                     '과거의 기록을 조회 중입니다.',
                     style: TextStyle(color: Colors.grey.shade400, fontSize: 12),
                   ),
                 ),
                 const SizedBox(height: 40),
              ],
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
      return Container(
        width: double.infinity,
        padding: const EdgeInsets.all(20),
        decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(15)),
        child: const Center(child: Text("측정된 감정 데이터가 없습니다.", style: TextStyle(color: Colors.grey, fontSize: 13))),
      );
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
    if (_isLoadingDiary) {
      return Container(
        height: 100,
        alignment: Alignment.center,
        child: const CircularProgressIndicator(color: Colors.blue, strokeWidth: 2),
      );
    }
    final List<String> images = _imageUrls;

    return Column(
      children: [
        Container(
          height: 175,
          width: double.infinity,
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(15),
            color: Colors.grey[200],
            image: images.isNotEmpty
                ? DecorationImage(
                    image: NetworkImage(images[0]),
                    fit: BoxFit.cover,
                  )
                : const DecorationImage(image: NetworkImage('https://via.placeholder.com/600x400'), fit: BoxFit.cover),
          ),
          child: images.isNotEmpty
              ? Image.network(
                  images[0],
                  fit: BoxFit.cover,
                  errorBuilder: (ctx, err, st) => const Center(child: Icon(Icons.broken_image, color: Colors.grey, size: 40)),
                  width: 0, height: 0, // Hidden but triggers errorBuilder if needed
                )
              : null,
        ),
        const SizedBox(height: 12),
        GridView.builder(
          shrinkWrap: true,
          physics: const NeverScrollableScrollPhysics(),
          itemCount: 3, // 1~3번째 이미지 표시 (총 4개 중 첫번째는 위에 크게)
          gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
            crossAxisCount: 3,
            crossAxisSpacing: 8,
            mainAxisSpacing: 8,
            childAspectRatio: 1,
          ),
          itemBuilder: (context, index) {
            final imgIndex = index + 1;
            return Container(
              decoration: BoxDecoration(
                borderRadius: BorderRadius.circular(12),
                color: Colors.grey[200],
              ),
              child: ClipRRect(
                borderRadius: BorderRadius.circular(12),
                child: images.length > imgIndex
                    ? Image.network(
                        images[imgIndex],
                        fit: BoxFit.cover,
                        errorBuilder: (ctx, err, st) => const Center(child: Icon(Icons.broken_image, color: Colors.grey)),
                      )
                    : const Center(child: Icon(Icons.pets, color: Colors.grey)),
              ),
            );
          },
        ),
      ],
    );
  }

  Widget _buildAISummaryCard(String petName) {
    if (_isLoadingDiary) {
       return Container(
        height: 60,
        alignment: Alignment.center,
        child: const CircularProgressIndicator(strokeWidth: 2),
      );
    }
    return Container(
      padding: const EdgeInsets.all(16.0),
      decoration: BoxDecoration(color: const Color(0xFFF0F5FF), borderRadius: BorderRadius.circular(12)),
      child: Text(
        _diaryContent ?? '해당 날짜에 생성된 일기가 없습니다.',
        style: const TextStyle(color: Colors.blueGrey, fontSize: 13, height: 1.5),
      ),
    );
  }

  Widget _buildTeacherAdviceCard(String petName) {
    if (_isLoadingDiary) {
      return Container(
        height: 80,
        alignment: Alignment.center,
        child: const CircularProgressIndicator(color: Colors.indigo, strokeWidth: 2),
      );
    }
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
              _reportContent ?? '분석 결과, 이 날 $petName는 평소보다 $_avgMood한 상태를 많이 보였습니다. 보호자님의 따뜻한 관심이 필요합니다.',
              style: const TextStyle(color: Colors.white, fontSize: 12),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildSavedMemoCard() {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(16.0),
      decoration: BoxDecoration(
        color: Colors.orange.shade50,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: Colors.orange.shade100),
      ),
      child: Text(
        (_memoContent == null || _memoContent!.trim().isEmpty) ? '저장된 메모가 없습니다.' : _memoContent!,
        style: TextStyle(color: Colors.orange.shade900, fontSize: 13, height: 1.5),
      ),
    );
  }

  Widget _buildMemoField() {
    return TextField(
      controller: _memoController,
      maxLines: 4,
      style: const TextStyle(fontSize: 13),
      decoration: InputDecoration(
        hintText: '특이 사항을 기록하세요...',
        hintStyle: const TextStyle(fontSize: 13, color: Colors.grey),
        filled: true,
        fillColor: Colors.white,
        contentPadding: const EdgeInsets.all(16),
        border: OutlineInputBorder(borderRadius: BorderRadius.circular(12), borderSide: const BorderSide(color: Colors.black12)),
        enabledBorder: OutlineInputBorder(borderRadius: BorderRadius.circular(12), borderSide: const BorderSide(color: Colors.black12)),
        focusedBorder: OutlineInputBorder(borderRadius: BorderRadius.circular(12), borderSide: const BorderSide(color: Colors.orange, width: 1.5)),
      ),
    );
  }

  Widget _buildBottomButton(BuildContext context) {
    // 일기 내용이 존재하면 생성 대신 저장/수정 모드로 판단
    // 문구의 길이나 내용을 체크하여 보다 정확하게 판별
    bool hasDiary = _diaryContent != null && _diaryContent!.trim().length > 10;
    
    return SizedBox(
      width: double.infinity,
      height: 50,
      child: ElevatedButton(
        onPressed: _isSaving 
            ? null 
            : (hasDiary ? () => _saveMemo() : _saveDiaryAndGenerate),
        style: ElevatedButton.styleFrom(
          backgroundColor: hasDiary ? const Color(0xFF4EA46C) : Colors.orange,
          disabledBackgroundColor: hasDiary ? const Color(0xFF388E3C) : Colors.orange.shade800,
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
          elevation: 0,
        ),
        child: _isSaving
            ? const Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  SizedBox(width: 20, height: 20, child: CircularProgressIndicator(color: Colors.white, strokeWidth: 2)),
                  SizedBox(width: 12),
                  Text('처리 중...', style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold)),
                ],
              )
            : Text(
                hasDiary ? '보호자 메모 수정하기' : '기록 저장 및 일기 생성',
                style: const TextStyle(color: Colors.white, fontWeight: FontWeight.bold),
              ),
      ),
    );
  }
}