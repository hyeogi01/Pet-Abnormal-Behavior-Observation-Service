import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:http/http.dart' as http;
import 'package:flutter/foundation.dart' show kIsWeb;
import 'dart:convert';
import 'package:pet_diary/mainPage/examination_history.dart';
import 'package:pet_diary/config.dart';

class ExaminationPage extends StatefulWidget {
  final Map<String, dynamic>? petData;
  final String? userId;
  const ExaminationPage({super.key, this.petData, this.userId});

  @override
  State<ExaminationPage> createState() => _ExaminationPageState();
}

class _ExaminationPageState extends State<ExaminationPage> {
  final ImagePicker _picker = ImagePicker();

  List<XFile> _eyeImages = [];
  List<XFile> _skinImages = [];

  bool _isUploading = false;

  Future<void> _pickImages(String diseaseType) async {
    final picked = await _picker.pickMultiImage(limit: 5);
    if (picked.isEmpty) return;
    setState(() {
      if (diseaseType == 'eye') _eyeImages = picked;
      else _skinImages = picked;
    });
  }

  Future<void> _uploadAndAnalyze(String diseaseType, List<XFile> images) async {
    if (images.isEmpty) return;
    setState(() => _isUploading = true);

    String petType = widget.petData?['pet_type'] ?? 'unknown';

    var uri = Uri.parse('${Config.apiBaseUrl}/api/analyze-disease');
    var request = http.MultipartRequest('POST', uri);
    request.headers.addAll(Config.ngrokHeaders);

    request.fields['pet_type'] = petType;
    request.fields['disease_type'] = diseaseType;
    request.fields['user_id'] = widget.userId ?? 'unknown_user';

    for (final img in images) {
      request.files.add(http.MultipartFile.fromBytes(
        'files',
        await img.readAsBytes(),
        filename: img.name,
      ));
    }

    try {
      var streamedResponse = await request.send();
      var response = await http.Response.fromStream(streamedResponse);

      if (!mounted) return;

      if (response.statusCode == 200) {
        final decodedBody = utf8.decode(response.bodyBytes);
        final Map<String, dynamic> result = json.decode(decodedBody);

        if (result['status'] == 'success') {
          showModalBottomSheet(
            context: context,
            isScrollControlled: true,
            backgroundColor: Colors.transparent,
            builder: (BuildContext context) {
              String category = result['disease_category'].toString().contains('eye') ? '안구 질환' : '피부 질환';
              String diagnosis = result['diagnosis'] ?? '진단 결과 없음';
              double probability = result['probability'] != null ? double.parse(result['probability'].toString()) : 0.0;
              int imagesAnalyzed = result['images_analyzed'] ?? images.length;
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
                    const SizedBox(height: 8),
                    Text(
                      '사진 $imagesAnalyzed장 분석 결과',
                      style: TextStyle(fontSize: 12, color: Colors.grey[500]),
                    ),
                    const SizedBox(height: 16),
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
        SnackBar(content: Text('서버 연결 오류: $e'), backgroundColor: Colors.red),
      );
    } finally {
      if (mounted) setState(() => _isUploading = false);
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
            image: DecorationImage(image: AssetImage(imagePath), fit: BoxFit.cover),
          ),
          child: Align(
            alignment: Alignment.topRight,
            child: Container(
              margin: const EdgeInsets.all(2),
              decoration: BoxDecoration(color: isGood ? Colors.green : Colors.redAccent, shape: BoxShape.circle),
              child: Icon(isGood ? Icons.check : Icons.close, color: Colors.white, size: 10),
            ),
          ),
        ),
        const SizedBox(height: 4),
        Text(
          isGood ? '좋은 예시' : '나쁜 예시',
          style: TextStyle(fontSize: 11, color: isGood ? Colors.green : Colors.redAccent, fontWeight: FontWeight.bold),
        )
      ],
    );
  }

  Widget _buildImageThumbnail(XFile img) {
    return ClipRRect(
      borderRadius: BorderRadius.circular(8),
      child: kIsWeb
          ? Image.network(img.path, width: 70, height: 70, fit: BoxFit.cover)
          : Image.file(File(img.path), width: 70, height: 70, fit: BoxFit.cover),
    );
  }

  Widget _buildDiseaseSection(String title, String diseaseType, List<XFile> selectedImages, IconData placeholderIcon) {
    final hasImages = selectedImages.isNotEmpty;

    return Card(
      elevation: 2,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
      margin: const EdgeInsets.symmetric(vertical: 12),
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(title, style: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold, color: Colors.black87)),
            const SizedBox(height: 16),
            // 이미지 미리보기 영역
            if (hasImages)
              SizedBox(
                height: 80,
                child: ListView.builder(
                  scrollDirection: Axis.horizontal,
                  itemCount: selectedImages.length,
                  itemBuilder: (_, i) => Padding(
                    padding: const EdgeInsets.only(right: 8),
                    child: _buildImageThumbnail(selectedImages[i]),
                  ),
                ),
              )
            else
              Center(
                child: Container(
                  width: double.infinity,
                  height: 80,
                  decoration: BoxDecoration(
                    color: Colors.grey[100],
                    borderRadius: BorderRadius.circular(12),
                    border: Border.all(color: Colors.grey[300]!),
                  ),
                  child: Icon(placeholderIcon, size: 36, color: Colors.grey[400]),
                ),
              ),
            const SizedBox(height: 12),
            // 예시 이미지 + 안내 문구
            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                _buildExampleThumbnail(diseaseType == 'eye' ? 'assets/images/eye_good.png' : 'assets/images/skin_good.png', true),
                const SizedBox(width: 16),
                _buildExampleThumbnail(diseaseType == 'eye' ? 'assets/images/eye_bad.png' : 'assets/images/skin_bad.png', false),
                const SizedBox(width: 16),
                Expanded(
                  child: Text(
                    '환부를 다각도로 찍은 사진 1~5장을 선택하세요.',
                    style: TextStyle(fontSize: 11, color: Colors.grey[600]),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 16),
            // 사진 선택 버튼
            SizedBox(
              width: double.infinity,
              child: OutlinedButton.icon(
                onPressed: _isUploading ? null : () => _pickImages(diseaseType),
                icon: const Icon(Icons.photo_library_outlined),
                label: Text(hasImages ? '사진 다시 선택 (${selectedImages.length}장)' : '사진 선택하기 (최대 5장)'),
                style: OutlinedButton.styleFrom(
                  padding: const EdgeInsets.symmetric(vertical: 12),
                  shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                ),
              ),
            ),
            const SizedBox(height: 8),
            // 분석하기 버튼
            SizedBox(
              width: double.infinity,
              child: ElevatedButton.icon(
                onPressed: (hasImages && !_isUploading) ? () => _uploadAndAnalyze(diseaseType, selectedImages) : null,
                icon: _isUploading
                    ? const SizedBox(width: 16, height: 16, child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white))
                    : const Icon(Icons.biotech),
                label: Text(_isUploading ? '분석 중...' : '분석하기'),
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.green,
                  foregroundColor: Colors.white,
                  padding: const EdgeInsets.symmetric(vertical: 14),
                  shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    String petName = widget.petData?['pet_name'] ?? '반려동물';

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
                        Text('$petName의 걱정되는 부위가 있으신가요?',
                            style: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold, color: Colors.white)),
                        const SizedBox(height: 4),
                        const Text('사진을 업로드하여 안구 및 피부 질환을 AI로 검사받으세요.',
                            style: TextStyle(fontSize: 14, color: Colors.white)),
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
                const Text('진단 부위 선택', style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold, color: Colors.black87)),
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
                    decoration: BoxDecoration(color: Colors.green[50], borderRadius: BorderRadius.circular(20)),
                    child: Text('이전 기록 보기 →', style: TextStyle(color: Colors.green[700], fontSize: 13, fontWeight: FontWeight.bold)),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 12),
            _buildDiseaseSection('👁️ 안구 질환 (Eye Disease)', 'eye', _eyeImages, Icons.visibility),
            _buildDiseaseSection('🩹 피부 질환 (Skin Disease)', 'skin', _skinImages, Icons.healing),
          ],
        ),
      ),
    );
  }
}
