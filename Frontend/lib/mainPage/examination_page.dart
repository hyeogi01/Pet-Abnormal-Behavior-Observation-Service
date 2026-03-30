import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:http/http.dart' as http;
import 'package:flutter/foundation.dart' show kIsWeb;
import 'dart:convert';
import 'package:pet_diary/mainPage/examination_history.dart';

class ExaminationPage extends StatefulWidget {
  final Map<String, dynamic>? petData;
  final String? userId;
  const ExaminationPage({super.key, this.petData, this.userId});

  @override
  State<ExaminationPage> createState() => _ExaminationPageState();
}

class _ExaminationPageState extends State<ExaminationPage> {
  final ImagePicker _picker = ImagePicker();
  
  // For selecting images
  XFile? _eyeImage;
  XFile? _skinImage;
  
  bool _isUploading = false;

  Future<void> _pickAndUploadImage(String diseaseType) async {
    final XFile? pickedFile = await _picker.pickImage(source: ImageSource.gallery);
    if (pickedFile == null) return;

    setState(() {
      if (diseaseType == 'eye') _eyeImage = pickedFile;
      if (diseaseType == 'skin') _skinImage = pickedFile;
      _isUploading = true;
    });

    // 펫 타입 가져오기 (기본값 설정)
    String petType = widget.petData?['pet_type'] ?? 'unknown';

    // 백엔드 API 주소 (Docker 환경에서 로컬 접속)
    var uri = Uri.parse('http://localhost:8080/api/analyze-disease');
    var request = http.MultipartRequest('POST', uri);

    // AI 모델에게 함께 보낼 메타데이터: 펫 종류와 질환 종류 (안구/피부)
    request.fields['pet_type'] = petType;
    request.fields['disease_type'] = diseaseType; // 'eye' 또는 'skin'
    request.fields['user_id'] = widget.userId ?? 'unknown_user';

    // 웹 환경과 모바일(앱/데스크탑) 환경 호환을 위한 분기 처리
    if (kIsWeb) {
      request.files.add(http.MultipartFile.fromBytes(
        'file', 
        await pickedFile.readAsBytes(),
        filename: pickedFile.name,
      ));
    } else {
      request.files.add(await http.MultipartFile.fromPath('file', pickedFile.path));
    }

    try {
      // 서버 전송 실행
      var streamedResponse = await request.send();
      var response = await http.Response.fromStream(streamedResponse);
      
      if (!mounted) return;
      
      if (response.statusCode == 200) {
        // UTF-8로 디코딩
        final decodedBody = utf8.decode(response.bodyBytes);
        final Map<String, dynamic> result = json.decode(decodedBody);
        
        if (result['status'] == 'success') {
          // 바텀시트로 결과 표시
          showModalBottomSheet(
            context: context,
            isScrollControlled: true,
            backgroundColor: Colors.transparent,
            builder: (BuildContext context) {
              String category = result['disease_category'].toString().contains('eye') ? '안구 질환' : '피부 질환';
              String diagnosis = result['diagnosis'] ?? '진단 결과 없음';
              double probability = result['probability'] != null ? double.parse(result['probability'].toString()) : 0.0;
              bool isNormal = diagnosis.contains('정상');
              Color statusColor = isNormal ? Colors.green : Colors.orange;
              IconData statusIcon = isNormal ? Icons.check_circle : Icons.warning_rounded;

              return Container(
                decoration: const BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.only(topLeft: Radius.circular(24), topRight: Radius.circular(24)),
                ),
                padding: const EdgeInsets.all(24),
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Container(width: 40, height: 4, decoration: BoxDecoration(color: Colors.grey[300], borderRadius: BorderRadius.circular(2))),
                    const SizedBox(height: 20),
                    Row(
                      children: [
                        Icon(statusIcon, color: statusColor, size: 28),
                        const SizedBox(width: 10),
                        Text('AI $category 분석 완료', style: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
                      ],
                    ),
                    const SizedBox(height: 24),
                    Container(
                      padding: const EdgeInsets.all(16),
                      decoration: BoxDecoration(color: statusColor.withOpacity(0.1), borderRadius: BorderRadius.circular(16)),
                      child: Row(
                        children: [
                          Expanded(
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Text('발견된 증상', style: TextStyle(color: statusColor, fontSize: 13, fontWeight: FontWeight.bold)),
                                const SizedBox(height: 4),
                                Text(diagnosis, style: const TextStyle(fontSize: 16, fontWeight: FontWeight.bold, color: Colors.black87)),
                              ],
                            ),
                          ),
                          Container(
                            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                            decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(12)),
                            child: Column(
                              children: [
                                const Text('AI 확신도', style: TextStyle(color: Colors.grey, fontSize: 11)),
                                Text('${probability.toStringAsFixed(1)}%', style: TextStyle(color: statusColor, fontSize: 16, fontWeight: FontWeight.bold)),
                              ],
                            ),
                          )
                        ],
                      ),
                    ),
                    const SizedBox(height: 24),
                    if (!isNormal)
                      Container(
                        padding: const EdgeInsets.all(16),
                        decoration: BoxDecoration(color: Colors.grey[50], borderRadius: BorderRadius.circular(12), border: Border.all(color: Colors.grey[200]!)),
                        child: const Row(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Icon(Icons.info_outline, color: Colors.grey, size: 20),
                            SizedBox(width: 10),
                            Expanded(child: Text('해당 질환이 의심됩니다. 정확한 진단을 위해 가까운 동물병원에 방문하시어 수의사와 상담해보시는 것을 권장합니다.', style: TextStyle(fontSize: 13, color: Colors.black54, height: 1.4))),
                          ],
                        ),
                      )
                    else
                      Container(
                        padding: const EdgeInsets.all(16),
                        decoration: BoxDecoration(color: Colors.grey[50], borderRadius: BorderRadius.circular(12), border: Border.all(color: Colors.grey[200]!)),
                        child: const Row(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Icon(Icons.health_and_safety, color: Colors.green, size: 20),
                            SizedBox(width: 10),
                            Expanded(child: Text('현재 육안 상으로 특별한 이상 징후가 발견되지 않았습니다. 지속적으로 관찰해주세요.', style: TextStyle(fontSize: 13, color: Colors.black54, height: 1.4))),
                          ],
                        ),
                      ),
                    const SizedBox(height: 32),
                    SizedBox(
                      width: double.infinity,
                      height: 50,
                      child: ElevatedButton(
                        onPressed: () => Navigator.of(context).pop(),
                        style: ElevatedButton.styleFrom(
                          backgroundColor: Colors.black87,
                          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                        ),
                        child: const Text('확인 완료', style: TextStyle(color: Colors.white, fontSize: 16, fontWeight: FontWeight.bold)),
                      ),
                    ),
                    const SizedBox(height: 16),
                  ],
                ),
              );
            },
          );
        } else {
          // 서버에서 에러 응답
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(content: Text('분석 실패: ${result['message']}')),
          );
        }
      } else {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('서버 에러: ${response.statusCode}')),
        );
      }
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('서버 연결 오류: $e'),
          backgroundColor: Colors.red,
        ),
      );
    } finally {
      if (mounted) {
        setState(() => _isUploading = false);
      }
    }
  }

  Widget _buildExampleThumbnail(String imagePath, bool isGood) {
    return Column(
      children: [
        Container(
          width: 50,
          height: 50,
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(8),
            border: Border.all(color: isGood ? Colors.green : Colors.redAccent, width: 2),
            image: DecorationImage(
              image: AssetImage(imagePath),
              fit: BoxFit.cover,
            ),
          ),
          child: Align(
            alignment: Alignment.topRight,
            child: Container(
              margin: const EdgeInsets.all(2),
              decoration: BoxDecoration(
                color: isGood ? Colors.green : Colors.redAccent,
                shape: BoxShape.circle,
              ),
              child: Icon(
                isGood ? Icons.check : Icons.close,
                color: Colors.white,
                size: 10,
              ),
            ),
          ),
        ),
        const SizedBox(height: 4),
        Text(
          isGood ? '좋은 예시' : '나쁜 예시',
          style: TextStyle(
            fontSize: 11,
            color: isGood ? Colors.green : Colors.redAccent,
            fontWeight: FontWeight.bold,
          ),
        )
      ],
    );
  }

  Widget _buildDiseaseSection(String title, String diseaseType, XFile? selectedImage, IconData placeholderIcon) {
    return Card(
      elevation: 2,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
      margin: const EdgeInsets.symmetric(vertical: 12),
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              title, 
              style: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold, color: Colors.black87)
            ),
            const SizedBox(height: 16),
            Row(
              children: [
                Expanded(
                  flex: 4,
                  child: AspectRatio(
                    aspectRatio: 1,
                    child: Container(
                      decoration: BoxDecoration(
                        color: Colors.grey[100],
                        borderRadius: BorderRadius.circular(15),
                        border: Border.all(color: Colors.grey[300]!),
                      ),
                      clipBehavior: Clip.hardEdge,
                      child: selectedImage != null 
                        ? (kIsWeb 
                            ? Image.network(selectedImage.path, fit: BoxFit.cover) 
                            : Image.file(File(selectedImage.path), fit: BoxFit.cover)) // 앱 환경일 때
                        : Center(child: Icon(placeholderIcon, size: 40, color: Colors.grey[400])),
                    ),
                  ),
                ),
                const SizedBox(width: 16),
                Expanded(
                  flex: 6,
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.stretch,
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      const Text(
                        '환부의 사진을 밝고 선명하게 찍어 업로드해주세요.',
                        style: TextStyle(fontSize: 11, color: Colors.grey),
                        textAlign: TextAlign.center,
                      ),
                      const SizedBox(height: 12),
                      Row(
                        mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                        children: [
                          _buildExampleThumbnail(diseaseType == 'eye' ? 'assets/images/eye_good.png' : 'assets/images/skin_good.png', true),
                          _buildExampleThumbnail(diseaseType == 'eye' ? 'assets/images/eye_bad.png' : 'assets/images/skin_bad.png', false),
                        ],
                      ),
                      const SizedBox(height: 20),
                      ElevatedButton.icon(
                        onPressed: _isUploading ? null : () => _pickAndUploadImage(diseaseType),
                        icon: _isUploading 
                          ? const SizedBox(width: 16, height: 16, child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white))
                          : const Icon(Icons.upload_file),
                        label: Text(_isUploading ? '분석 중...' : '사진 올리고 분석하기'),
                        style: ElevatedButton.styleFrom(
                          backgroundColor: Colors.green,
                          foregroundColor: Colors.white,
                          padding: const EdgeInsets.symmetric(vertical: 16),
                          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                        ),
                      )
                    ],
                  ),
                ),
              ],
            )
          ],
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    String petName = widget.petData?['pet_name'] ?? '반려동물';
    String petType = widget.petData?['pet_type'] ?? '정보없음';
    
    return Scaffold(
      backgroundColor: Colors.white,
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(20.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Container(
              padding: const EdgeInsets.all(20),
              decoration: BoxDecoration(
                borderRadius: BorderRadius.circular(20),
                gradient: LinearGradient(colors: [Colors.green[400]!, Colors.teal[400]!]),
              ),
              child: Row(
                children: [
                  const Icon(Icons.health_and_safety, color: Colors.white, size: 40),
                  const SizedBox(width: 16),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        //Text('AI 질환 검진소', style: const TextStyle(color: Colors.white, fontSize: 20, fontWeight: FontWeight.bold)),
                        Text('$petName의 걱정되는 부위가 있으신가요?',
              style: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold, color: Colors.white),
            ),
                        const SizedBox(height: 4),
                        //Text('현재 종: $petType', style: const TextStyle(color: Colors.white70, fontSize: 13)),
                        const Text(
              '사진을 업로드하여 안구 및 피부 질환을 AI로 검사받으세요.',
              style: TextStyle(fontSize: 14, color: Colors.white),
            ),
                      ],
                    ),
                  ),
                ],
              ),
            ),
            
            const SizedBox(height: 24),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                const Text(
                  '진단 부위 선택',
                  style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold, color: Colors.black87),
                ),
                GestureDetector(
                  onTap: () {
                    Navigator.push(
                      context,
                      MaterialPageRoute(
                        builder: (context) => ExaminationHistoryPage(userId: widget.userId ?? 'test_user', petData: widget.petData),
                      ),
                    );
                  },
                  child: Container(
                    padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                    decoration: BoxDecoration(
                      color: Colors.green[50],
                      borderRadius: BorderRadius.circular(20),
                    ),
                    child: Text(
                      '이전 기록 보기 →',
                      style: TextStyle(color: Colors.green[700], fontSize: 13, fontWeight: FontWeight.bold),
                    ),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 12),
            _buildDiseaseSection('👁️ 안구 질환 (Eye Disease)', 'eye', _eyeImage, Icons.visibility),
            _buildDiseaseSection('🩹 피부 질환 (Skin Disease)', 'skin', _skinImage, Icons.healing),
          ],
        ),
      ),
    );
  }
}
