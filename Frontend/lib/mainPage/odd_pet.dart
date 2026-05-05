import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'package:image_picker/image_picker.dart';

class PageB extends StatefulWidget {
  final String userId;
  final Map<String, dynamic>? petData;
  const PageB({super.key, required this.userId, this.petData});

  @override
  State<PageB> createState() => _PageBState();
}

class _PageBState extends State<PageB> {
  bool _isAnalyzing = false;
  String? _analysisResult;

  // 실제 데이터 연동을 위한 상태 변수 (초기값 설정)
  String? _lastDetectionTime; // 마지막 감지 시간
  String? _detectionImageUrl; // 감지된 이미지 URL
  double _aiConfidence = 0.0; // AI 신뢰도
  int _patellarHealthScore = 0; // 슬개골 건강도 점수
  int _abnormalCount = 0; // 이상 감지 건수
  int _totalCount = 0; // 전체 분석 이미지 수
  int _maxSeverity = 0; // 최고 위험도 등급 (1~4)

  bool _isLoadingData = true;

  @override
  void initState() {
    super.initState();
    _initializeData();
  }

  Future<void> _initializeData() async {
    // 1. Pet Info (슬개골 건강도) 초기화 (데이터 없을 시 기본값 100)
    _patellarHealthScore = 100;

    // 2. Day Data (오늘 이상 행동 및 슬개골 판별) - 백엔드 API 호출
    await _fetchDayLogs();

    setState(() {
      _isLoadingData = false;
    });
  }

  Future<void> _fetchDayLogs() async {
    final now = DateTime.now();
    final todayStr = '${now.year}-${now.month.toString().padLeft(2, '0')}-${now.day.toString().padLeft(2, '0')}';
    final url = Uri.parse('http://localhost:8080/api/day-logs/${widget.userId}?date=$todayStr');
    print('[DEBUG] Fetching day logs: userId=${widget.userId}, date=$todayStr');

    try {
      final response = await http.get(url);
      if (response.statusCode == 200) {
        final decoded = jsonDecode(response.body);
        if (decoded['status'] == 'success' && decoded['data'] != null) {
          final data = Map<String, dynamic>.from(decoded['data']);
          if (data.isNotEmpty) {
            final sortedKeys = data.keys.toList()..sort();
            
            int abnormalCount = 0;
            int totalCount = 0;
            int maxSeverity = 0;
            
            String? latestAbnormalTimeKey;
            Map<String, dynamic>? latestAbnormalData;
            double? latestAbnormalConfidence;
            
            // 1. 모든 로그를 순회하며 이상 행동 횟수, 최고 위험도(낮은 점수), 가장 최근 이상 기록을 찾습니다.
            for (var key in sortedKeys) {
              final logData = Map<String, dynamic>.from(data[key] as Map);
              if (logData['analysis_result'] != null) {
                final analysisResult = Map<String, dynamic>.from(logData['analysis_result'] as Map);
                if (analysisResult['patella_analysis'] != null) {
                  final patella = Map<String, dynamic>.from(analysisResult['patella_analysis'] as Map);
                  final gradeStr = patella['status']?.toString();
                  totalCount++; // patella_analysis가 있는 모든 이미지 카운트
                  
                  if (gradeStr != null && gradeStr != 'normal') {
                    abnormalCount++; // 이상 건수 증가
                    
                    int grade = int.tryParse(gradeStr) ?? 0;
                    if (grade > maxSeverity) {
                      maxSeverity = grade; // 가장 심각한 단계(1~4) 갱신
                    }
                    
                    // sortedKeys가 시간순이므로 마지막으로 발견된 것이 '가장 최신'입니다.
                    latestAbnormalTimeKey = key;
                    latestAbnormalData = logData;
                    latestAbnormalConfidence = (patella['confidence'] as num?)?.toDouble() ?? 0.85;
                  }
                }
              }
            }
            
            // 2. 이상 로그가 있는지 확인하여 상태를 업데이트합니다.
            if (latestAbnormalData != null) {
              setState(() {
                _lastDetectionTime = latestAbnormalTimeKey.toString().length >= 5 ? latestAbnormalTimeKey.toString().substring(0, 5) : latestAbnormalTimeKey.toString();
                _detectionImageUrl = latestAbnormalData!['image_url'];
                _aiConfidence = latestAbnormalConfidence ?? 0.85;
                _patellarHealthScore = 100 - (maxSeverity * 20); // 가장 심각한 등급으로 점수 산정
                _abnormalCount = abnormalCount;
                _totalCount = totalCount;
                _maxSeverity = maxSeverity;
                _analysisResult = null; // 결과 초기화
              });
            } else {
              // 이상 로그가 전혀 없는 경우: 이미지를 보여주지 않고 안내 문구 표시
              setState(() {
                _lastDetectionTime = null;
                _detectionImageUrl = null;
                _aiConfidence = 0.0;
                _patellarHealthScore = 100;
                _abnormalCount = 0;
                _totalCount = totalCount;
                _maxSeverity = 0;
                _analysisResult = '오늘 감지된 이상 행동이 없습니다.';
              });
            }
            return;
          }
        }
      }
    } catch (e) {
      print('Failed to fetch day logs: $e');
      setState(() {
        _lastDetectionTime = null;
        _detectionImageUrl = null;
        _aiConfidence = 0.0;
        _patellarHealthScore = 100;
        _abnormalCount = 0;
        _analysisResult = 'Error: $e'; // 에러를 UI에 띄워줍니다
      });
      return;
    }
    
    // 실패했거나 데이터가 없는 경우
    setState(() {
      _lastDetectionTime = null;
      _detectionImageUrl = null;
      _aiConfidence = 0.0;       // 기본값
      _patellarHealthScore = 100; // 기본값
      _abnormalCount = 0;
      if (_analysisResult == null) {
        _analysisResult = '데이터가 없습니다. (요청일: $todayStr)';
      }
    });
  }

  // 실제 파일 업로드 및 AI 분석 함수
  Future<void> _uploadAndAnalyze({required bool isVideo}) async {
    final picker = ImagePicker();
    XFile? pickedFile;

    try {
      if (isVideo) {
        pickedFile = await picker.pickVideo(source: ImageSource.gallery);
      } else {
        pickedFile = await picker.pickImage(source: ImageSource.gallery);
      }
    } catch (e) {
      setState(() {
        _analysisResult = '파일 선택 오류: $e';
      });
      return;
    }

    if (pickedFile == null) return; // 사용자가 취소한 경우

    setState(() {
      _isAnalyzing = true;
      _analysisResult = null;
    });

    try {
      final bytes = await pickedFile.readAsBytes();
      final petType = widget.petData?['pet_type'] ?? 'dog';

      final request = http.MultipartRequest(
        'POST',
        Uri.parse('http://localhost:8080/api/analyze-patella/${widget.userId}'),
      );
      request.fields['pet_type'] = petType.toString().toLowerCase();
      request.files.add(http.MultipartFile.fromBytes(
        'file',
        bytes,
        filename: pickedFile.name,
      ));

      final streamedResponse = await request.send();
      final response = await http.Response.fromStream(streamedResponse);

      if (response.statusCode == 200) {
        final decoded = jsonDecode(response.body);
        if (decoded['status'] == 'success') {
          final gradeText = decoded['grade_text'] ?? '결과 없음';
          final confidence = ((decoded['patella_confidence'] as num?) ?? 0.0) * 100;
          final fileType = decoded['file_type'] == 'video' ? '동영상' : '사진';
          setState(() {
            _isAnalyzing = false;
            _analysisResult = '[$fileType 분석 완료] $gradeText\nAI 신뢰도: ${confidence.toStringAsFixed(1)}%';
          });
        } else {
          setState(() {
            _isAnalyzing = false;
            _analysisResult = '분석 실패: ${decoded['message'] ?? '알 수 없는 오류'}';
          });
        }
      } else {
        setState(() {
          _isAnalyzing = false;
          _analysisResult = '서버 오류 (HTTP ${response.statusCode})';
        });
      }
    } catch (e) {
      setState(() {
        _isAnalyzing = false;
        _analysisResult = '업로드 오류: $e';
      });
    }
  }


  @override
  Widget build(BuildContext context) {
    if (_isLoadingData) {
      return const Scaffold(
        body: Center(child: CircularProgressIndicator(color: Colors.orange)),
      );
    }
    return Scaffold(
      backgroundColor: const Color(0xFFF8F8F8),
      body: Column(
        children: [
          // 1. 상단 주황색 헤더 영역
          _buildHeader(context),

          Expanded(
            child: SingleChildScrollView(
              padding: const EdgeInsets.all(16.0),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  // 2. 오늘 감지된 이상 행동 섹션
                  _buildSectionTitle(Icons.warning_amber_rounded, '오늘 감지된 이상 행동', badgeCount: '${_abnormalCount}건'),
                  const SizedBox(height: 12),
                  _buildBehaviorCard(
                    time: _lastDetectionTime,
                    title: '슬개골 관련',
                    description: _detectionImageUrl != null 
                        ? '슬개골 이상 및 보행 절뚝거림이 탐지되었습니다.' 
                        : '현재 탐지된 슬개골 이상 보행이 없습니다.',
                    confidence: _aiConfidence,
                    color: Colors.orange,
                    imageUrl: _detectionImageUrl,
                  ),
                  const SizedBox(height: 12),
                  _buildBehaviorCard(
                    time: _lastDetectionTime,
                    title: '슬개골 이상',
                    description: _maxSeverity > 0
                        ? '슬개골 질환 ${_maxSeverity}기가 의심됩니다'
                        : '슬개골 이상 없음',
                    confidenceLabel: '슬개골 확인 빈도',
                    confidence: _totalCount > 0 ? _abnormalCount / _totalCount : 0.0,
                    color: Colors.purple,
                    imageUrl: null,
                  ),

                  const SizedBox(height: 24),

                  // 3. 건강 지표 모니터링 섹션
                  _buildSectionTitle(Icons.favorite_border, '건강 지표 모니터링'),
                  const SizedBox(height: 12),
                  _buildHealthIndicatorBox(),

                  const SizedBox(height: 24),

                  // 4. 슬개골 건강 분석 섹션 (업로드 및 AI 판별 기능 포함)
                  _buildSectionTitle(Icons.analytics_outlined, '슬개골 건강 분석'),
                  const SizedBox(height: 12),
                  _buildAnalysisCard(),

                  const SizedBox(height: 24),

                  // 5. AI 분석 기술 설명 섹션
                  _buildAITechInfo(),

                  const SizedBox(height: 24),

                  // 6. 일기 저장하기 버튼
                  _buildSaveButton(),
                  const SizedBox(height: 40),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  // --- 위젯 빌더 함수들 ---

  Widget _buildHeader(BuildContext context) {
    final now = DateTime.now();
    final weekdays = ['월요일', '화요일', '수요일', '목요일', '금요일', '토요일', '일요일'];
    final dateStr = '${now.year}년 ${now.month}월 ${now.day}일 ${weekdays[now.weekday - 1]}';

    return AppBar(
      backgroundColor: Colors.orange,
      elevation: 0,
      leading: IconButton(
        icon: const Icon(Icons.arrow_back, color: Colors.white, size: 20),
        onPressed: () => Navigator.pop(context),
      ),
      title: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          const Text(
            '이상 행동 일기',
            style: TextStyle(color: Colors.white, fontSize: 16, fontWeight: FontWeight.bold),
          ),
          Text(
            dateStr,
            style: const TextStyle(color: Colors.white70, fontSize: 10),
          ),
        ],
      ),
      centerTitle: true,
    );
  }

  Widget _buildSectionTitle(IconData icon, String title, {String? badgeCount}) {
    return Row(
      children: [
        Icon(icon, color: Colors.redAccent, size: 22),
        const SizedBox(width: 8),
        Text(title, style: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
        if (badgeCount != null) ...[
          const Spacer(),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 2),
            decoration: BoxDecoration(color: Colors.orange[100], borderRadius: BorderRadius.circular(10)),
            child: Text(badgeCount, style: const TextStyle(color: Colors.orange, fontSize: 12, fontWeight: FontWeight.bold)),
          ),
        ]
      ],
    );
  }

  Widget _buildBehaviorCard({
    String? time,
    required String title,
    required String description,
    required double confidence,
    required Color color,
    String? imageUrl,
    String confidenceLabel = 'AI 신뢰도',
  }) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(15),
        border: Border.all(color: color.withOpacity(0.3)),
        boxShadow: const [BoxShadow(color: Colors.black12, blurRadius: 4)],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              if (time != null && time.isNotEmpty) ...[
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
                  decoration: BoxDecoration(color: color, borderRadius: BorderRadius.circular(4)),
                  child: Text(time, style: const TextStyle(color: Colors.white, fontSize: 11, fontWeight: FontWeight.bold)),
                ),
                const SizedBox(width: 8),
              ],
              Text(title, style: TextStyle(color: color, fontWeight: FontWeight.bold, fontSize: 16)),
            ],
          ),
          const SizedBox(height: 8),
          Text(description, style: const TextStyle(fontSize: 13, color: Colors.black87)),
          if (imageUrl != null && imageUrl.isNotEmpty) ...[
            const SizedBox(height: 12),
            ClipRRect(
              borderRadius: BorderRadius.circular(10),
              child: Image.network(
                imageUrl,
                height: 200,
                width: double.infinity,
                fit: BoxFit.cover,
                errorBuilder: (context, error, stackTrace) => Container(
                  height: 200,
                  width: double.infinity,
                  color: Colors.grey[200],
                  child: const Center(child: Icon(Icons.broken_image, color: Colors.grey)),
                ),
              ),
            ),
          ],
          const SizedBox(height: 12),
          Row(
            children: [
              Text(confidenceLabel, style: const TextStyle(fontSize: 12, color: Colors.grey)),
              const SizedBox(width: 8),
              Expanded(
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(10),
                  child: LinearProgressIndicator(
                    value: confidence,
                    backgroundColor: Colors.grey[200],
                    valueColor: AlwaysStoppedAnimation<Color>(confidence > 0 ? color : Colors.grey),
                    minHeight: 8,
                  ),
                ),
              ),
              const SizedBox(width: 10),
              Text('${(confidence * 100).toInt()}%', style: TextStyle(fontSize: 12, color: confidence > 0 ? color : Colors.grey, fontWeight: FontWeight.bold)),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildHealthIndicatorBox() {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(15),
        boxShadow: const [BoxShadow(color: Colors.black12, blurRadius: 4)],
      ),
      child: Column(
        children: [
          _buildIndicatorRow(
            '슬개골 건강도', 
            _patellarHealthScore, 
            _patellarHealthScore > 0 ? Colors.orange : Colors.grey, 
            _getPatellaStatusText()
          ),
        ],
      ),
    );
  }

  Widget _buildIndicatorRow(String label, int value, Color color, String subText) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Text(label, style: const TextStyle(fontWeight: FontWeight.bold)),
            Text('$value', style: TextStyle(color: color, fontSize: 18, fontWeight: FontWeight.bold)),
          ],
        ),
        const SizedBox(height: 8),
        ClipRRect(
          borderRadius: BorderRadius.circular(10),
          child: LinearProgressIndicator(
            value: value / 100,
            backgroundColor: Colors.grey[100],
            valueColor: AlwaysStoppedAnimation<Color>(color),
            minHeight: 10,
          ),
        ),
        const SizedBox(height: 4),
        Text(subText, style: const TextStyle(fontSize: 11, color: Colors.grey)),
      ],
    );
  }

  Widget _buildAnalysisCard() {
    return Container(
      width: double.infinity,
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(15),
        boxShadow: const [BoxShadow(color: Colors.black12, blurRadius: 4)],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Container(
            padding: const EdgeInsets.all(16),
            decoration: BoxDecoration(color: Colors.orange[50], borderRadius: const BorderRadius.vertical(top: Radius.circular(15))),
            child: Row(
              children: [
                const Icon(Icons.camera_alt, color: Colors.orange, size: 24),
                const SizedBox(width: 10),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Text('슬개골 건강 AI 분석', style: TextStyle(fontWeight: FontWeight.bold, fontSize: 15)),
                      const SizedBox(height: 4),
                      const Text('동영상이나 사진을 업로드하여 슬개골 이상 여부를 판별해보세요.', style: TextStyle(fontSize: 12, height: 1.5, color: Colors.black54)),
                    ],
                  ),
                )
              ],
            ),
          ),
          Padding(
            padding: const EdgeInsets.all(16.0),
            child: _isAnalyzing 
              ? const Center(
                  child: Padding(
                    padding: EdgeInsets.symmetric(vertical: 20.0),
                    child: Column(
                      children: [
                        CircularProgressIndicator(color: Colors.orange),
                        SizedBox(height: 16),
                        Text('AI 모델이 슬개골 상태를 분석 중입니다...', style: TextStyle(fontSize: 12, color: Colors.grey)),
                      ],
                    ),
                  ),
                )
              : _analysisResult == null 
                  ? Row(
                      children: [
                        Expanded(
                          child: ElevatedButton.icon(
                            onPressed: () => _uploadAndAnalyze(isVideo: true),
                            icon: const Icon(Icons.video_camera_back, size: 18),
                            label: const Text('동영상 업로드'),
                            style: ElevatedButton.styleFrom(
                              foregroundColor: Colors.white,
                              backgroundColor: Colors.orangeAccent,
                              elevation: 0,
                              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
                              padding: const EdgeInsets.symmetric(vertical: 12),
                            ),
                          ),
                        ),
                        const SizedBox(width: 12),
                        Expanded(
                          child: ElevatedButton.icon(
                            onPressed: () => _uploadAndAnalyze(isVideo: false),
                            icon: const Icon(Icons.image, size: 18),
                            label: const Text('사진 업로드'),
                            style: ElevatedButton.styleFrom(
                              foregroundColor: Colors.white,
                              backgroundColor: Colors.blueAccent,
                              elevation: 0,
                              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
                              padding: const EdgeInsets.symmetric(vertical: 12),
                            ),
                          ),
                        ),
                      ],
                    )
                  : Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Container(
                          padding: const EdgeInsets.all(12),
                          decoration: BoxDecoration(
                            color: Colors.orange[50],
                            borderRadius: BorderRadius.circular(8),
                            border: Border.all(color: Colors.orange.withOpacity(0.3)),
                          ),
                          child: Row(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Icon(Icons.warning_rounded, color: Colors.orange[400], size: 20),
                              const SizedBox(width: 8),
                              Expanded(
                                child: Text(
                                  _analysisResult!,
                                  style: TextStyle(fontSize: 13, color: Colors.orange[800], height: 1.5, fontWeight: FontWeight.w600),
                                ),
                              ),
                            ],
                          ),
                        ),
                        const SizedBox(height: 16),
                        const Row(
                          children: [
                            CircleAvatar(radius: 12, backgroundColor: Colors.blueAccent, child: Icon(Icons.medical_services, color: Colors.white, size: 14)),
                            SizedBox(width: 8),
                            Text('수의사 권장사항', style: TextStyle(fontWeight: FontWeight.bold, fontSize: 14)),
                          ],
                        ),
                        const SizedBox(height: 8),
                        const Text('• 무리한 점프나 계단 오르내리기를 줄여주세요', style: TextStyle(fontSize: 12, height: 1.6)),
                        const Text('• 체중 관리를 통해 관절 부담을 줄여주세요', style: TextStyle(fontSize: 12, height: 1.6)),
                        const Text('• 증상이 지속되면 동물병원 방문을 권장합니다', style: TextStyle(fontSize: 12, height: 1.6)),
                        const SizedBox(height: 16),
                        Center(
                          child: TextButton.icon(
                            onPressed: () {
                              setState(() {
                                _analysisResult = null;
                              });
                            },
                            icon: const Icon(Icons.refresh, size: 16),
                            label: const Text('다시 분석하기'),
                          ),
                        )
                      ],
                    ),
          ),
        ],
      ),
    );
  }

  Widget _buildAITechInfo() {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        gradient: const LinearGradient(colors: [Color(0xFF8E2DE2), Color(0xFFF64C75)]),
        borderRadius: BorderRadius.circular(15),
      ),
      child: const Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(Icons.description_outlined, color: Colors.white, size: 20),
              SizedBox(width: 8),
              Text('AI 분석 기술', style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold)),
            ],
          ),
          SizedBox(height: 8),
          Text(
            '이 일기는 8가지 데이터셋을 활용한 AI 분석으로 작성됩니다. 반려동물 보행영상, 행동 데이터, 건강정보, 관절 인식, 안구/피부 증상 이미지, X-ray 분석 등을 통해 콩이의 건강을 24시간 모니터링합니다.',
            style: TextStyle(color: Colors.white, fontSize: 11, height: 1.5),
          ),
        ],
      ),
    );
  }

  Widget _buildSaveButton() {
    return SizedBox(
      width: double.infinity,
      height: 55,
      child: ElevatedButton(
        onPressed: () {},
        style: ElevatedButton.styleFrom(
          backgroundColor: Colors.orange,
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
          elevation: 2,
        ),
        child: const Text('일기 저장하기', style: TextStyle(color: Colors.white, fontSize: 18, fontWeight: FontWeight.bold)),
      ),
    );
  }

  String _getPatellaStatusText() {
    if (_totalCount == 0) return '데이터 수집 및 AI 분석이 진행 중입니다.';
    switch (_maxSeverity) {
      case 1:
        return '가벼운 주의가 필요합니다.';
      case 2:
        return '세심한 관찰이 필요합니다.';
      case 3:
        return '전문가의 상담을 권장합니다.';
      case 4:
        return '즉각적인 전문 진료가 시급합니다.';
      default:
        return '슬개골 건강 상태가 양호하며 정상 보행을 보입니다.';
    }
  }
}
