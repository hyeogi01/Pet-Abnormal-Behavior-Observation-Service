import 'package:flutter/material.dart';
import 'package:pet_diary/main.dart';
import 'package:http/http.dart' as http; // 상단에 추가
import 'dart:convert'; // JSON 변환을 위해 추가

class PetRegistrationPage extends StatefulWidget {
  final String petName;
  final String userId; // [추가] 로그인한 유저의 ID를 저장할 변수

  const PetRegistrationPage({
    Key? key,
    required this.petName,
    required this.userId // [추가] 필수 인자로 설정
  }) : super(key: key);

  @override
  _PetRegistrationPageState createState() => _PetRegistrationPageState();
}

class _PetRegistrationPageState extends State<PetRegistrationPage> {
  // 상태 변수들
  String _selectedSpecies = '강아지';
  String _selectedGender = '남아';
  final TextEditingController _speciesDetailController = TextEditingController();
  DateTime? _birthDate;

  String _neuteredStatus = '안했어요';
  final TextEditingController _diseaseController = TextEditingController();
  String _separationAnxiety = '모르겠어요';

  // 사용자 입력 정보를 DB에 저장하는 함수
  Future<void> _savePetInfo() async {
    final Uri url = Uri.parse('http://localhost:8000/user-input/${widget.userId}');

    try {
      final response = await http.post(
        url,
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
          'pet_name': widget.petName,                                    // 이전 페이지에서 넘어온 이름
          'pet_type': '${_selectedSpecies} (${_speciesDetailController.text})', // 종 + 상세품종
          'pet_gender': _selectedGender,                                 // 성별
          'pet_birthday': _birthDate != null
              ? '${_birthDate!.year}-${_birthDate!.month.toString().padLeft(2, '0')}-${_birthDate!.day.toString().padLeft(2, '0')}'
              : '',                                                      // 생년월일 문자열 포맷
        }),
      );

      if (response.statusCode == 200) {
        print('DB 저장 성공!');
      } else {
        print('저장 실패: ${response.statusCode} - ${response.body}');
      }
    } catch (e) {
      print('네트워크 에러: $e');
    }
  }
  // [추가] 필수 입력값이 채워졌는지 확인하는 Getter
  bool get _isFormValid {
    return _speciesDetailController.text.trim().isNotEmpty && _birthDate != null;
  }

  @override
  void initState() {
    super.initState();
    // [추가] 텍스트 입력 시마다 버튼 상태를 새로고침하기 위한 리스너
    _speciesDetailController.addListener(_onTextFieldChanged);
  }

  @override
  void dispose() {
    _speciesDetailController.removeListener(_onTextFieldChanged);
    _speciesDetailController.dispose();
    _diseaseController.dispose();
    super.dispose();
  }

  void _onTextFieldChanged() {
    setState(() {}); // 텍스트 변경 시 build() 재실행하여 버튼 색상 업데이트
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.white,
      appBar: AppBar(
        backgroundColor: Colors.white,
        elevation: 0,
        leading: IconButton(
          icon: const Icon(Icons.arrow_back_ios, color: Colors.black, size: 20),
          onPressed: () {
            // 현재 페이지를 제거하고 OnboardingPage3로 완전히 교체합니다
          },
        ),
        title: const Text('반려동물 등록',
            style: TextStyle(color: Colors.black, fontSize: 16, fontWeight: FontWeight.bold)),
        centerTitle: true,
        bottom: PreferredSize(
          preferredSize: const Size.fromHeight(4.0),
          child: LinearProgressIndicator(
            value: 1.0,
            backgroundColor: Colors.grey[200],
            valueColor: const AlwaysStoppedAnimation<Color>(Colors.orange),
            minHeight: 2,
          ),
        ),
      ),
      body: SingleChildScrollView(
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 32),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Text('궁금해요', style: TextStyle(fontSize: 16, color: Colors.black54)),
              const SizedBox(height: 8),
              RichText(
                text: TextSpan(
                  children: [
                    TextSpan(
                      text: '${widget.petName}에 대해\n더 알려 주실래요? 🐶',
                      style: const TextStyle(fontSize: 26, fontWeight: FontWeight.bold, color: Colors.black, height: 1.3),
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 24),
              _buildAiGuideBox(),
              const SizedBox(height: 40),

              // 1. 종 선택
              _buildSectionTitle('종'),
              const SizedBox(height: 12),
              Row(
                children: [
                  _buildSpeciesCard('강아지', Icons.pets),
                  const SizedBox(width: 12),
                  _buildSpeciesCard('고양이', Icons.catching_pokemon),
                  const SizedBox(width: 12),
                  _buildSpeciesCard('다른 동물', Icons.emoji_nature),
                ],
              ),
              const SizedBox(height: 12),
              TextField(
                controller: _speciesDetailController,
                decoration: _inputDecoration('품종을 입력해주세요 (예: 말티즈)'),
              ),
              const SizedBox(height: 40),

              // 2. 성별 선택
              _buildSectionTitle('성별'),
              const SizedBox(height: 12),
              Row(
                children: [
                  Expanded(child: _buildGenderCard('남아', Icons.male)),
                  const SizedBox(width: 12),
                  Expanded(child: _buildGenderCard('여아', Icons.female)),
                ],
              ),
              const SizedBox(height: 40),

              // 3. 생년월일
              _buildSectionTitle('생년월일'),
              const SizedBox(height: 12),
              GestureDetector(
                onTap: _selectDate,
                child: Container(
                  width: double.infinity,
                  padding: const EdgeInsets.symmetric(vertical: 16, horizontal: 16),
                  decoration: BoxDecoration(
                    color: Colors.white,
                    border: Border.all(color: Colors.grey[300]!),
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: Text(
                    _birthDate == null
                        ? '생년월일을 선택해주세요'
                        : '${_birthDate!.year}-${_birthDate!.month.toString().padLeft(2, '0')}-${_birthDate!.day.toString().padLeft(2, '0')}',
                    style: TextStyle(
                      color: _birthDate == null ? Colors.grey[400] : Colors.black,
                      fontSize: 16,
                    ),
                  ),
                ),
              ),
              const SizedBox(height: 40),

              // 4. 중성화 여부
              _buildSectionTitle('중성화 여부'),
              const SizedBox(height: 12),
              _buildCheckRadio('안했어요', _neuteredStatus, (val) => setState(() => _neuteredStatus = val)),
              const SizedBox(height: 10),
              _buildCheckRadio('중성화했어요', _neuteredStatus, (val) => setState(() => _neuteredStatus = val)),
              const SizedBox(height: 40),

              // 5. 앓고 있는 질환
              _buildSectionTitle('앓고 있는 질환'),
              const SizedBox(height: 12),
              TextField(
                controller: _diseaseController,
                decoration: _inputDecoration('앓고 있는 질환 정보를 알려주세요'),
              ),
              const SizedBox(height: 40),

              // 6. 분리불안 여부
              _buildSectionTitle('분리불안 여부'),
              const SizedBox(height: 12),
              _buildCircleRadio('있어요', _separationAnxiety, (val) => setState(() => _separationAnxiety = val)),
              _buildCircleRadio('없어요', _separationAnxiety, (val) => setState(() => _separationAnxiety = val)),
              _buildCircleRadio('모르겠어요', _separationAnxiety, (val) => setState(() => _separationAnxiety = val)),
              const SizedBox(height: 50),

              // --- 완료 버튼 (수정됨) ---
              SizedBox(
                width: double.infinity,
                height: 56,
                child: ElevatedButton(
                  onPressed: _isFormValid ? () {
                    // 모든 정보가 입력되었을 때 실행
                    _navigateToDashboard();
                  } : null, // 비활성화 상태
                  style: ElevatedButton.styleFrom(
                    backgroundColor: _isFormValid ? Colors.orange : Colors.grey[300],
                    disabledBackgroundColor: Colors.grey[300], // 비활성 시 색상 명시
                    elevation: 0,
                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                  ),
                  child: const Text('다음으로',
                      style: TextStyle(color: Colors.white, fontSize: 16, fontWeight: FontWeight.bold)),
                ),
              ),
              const SizedBox(height: 40),
            ],
          ),
        ),
      ),
    );
  }

  // --- 비즈니스 로직 ---

  Future<void> _selectDate() async {
    final DateTime? picked = await showDatePicker(
      context: context,
      initialDate: DateTime.now(),
      firstDate: DateTime(2000),
      lastDate: DateTime.now(),
      builder: (context, child) {
        return Theme(
          data: Theme.of(context).copyWith(
            colorScheme: const ColorScheme.light(primary: Colors.orange),
          ),
          child: child!,
        );
      },
    );
    if (picked != null) {
      setState(() => _birthDate = picked);
    }
  }

  void _navigateToDashboard() async{
    await _savePetInfo();

    // 2. 화면 이동
    if (!mounted) return;
    print('등록 완료: ${widget.petName}');
    // 로직 추가
    Navigator.pushReplacement(
      context,
      MaterialPageRoute(
        builder: (context) => PetHealthDashboard(userId: widget.userId),
      ),
    );

    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text('${widget.petName} 등록이 완료되었습니다!')),
    );

    // 다음 페이지로 이동 (PetHealthDashboard 클래스가 정의되어 있어야 함)
    // Navigator.push(
    //   context,
    //   MaterialPageRoute(builder: (context) => const PetHealthDashboard()),
    // );
  }

  // --- 위젯 헬퍼 함수들 ---

  Widget _buildAiGuideBox() {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.grey[100],
        borderRadius: BorderRadius.circular(12),
      ),
      child: Row(
        children: [
          const Icon(Icons.pets, color: Colors.orange, size: 20),
          const SizedBox(width: 10),
          Expanded(
            child: Text(
              '${widget.petName} 정보를 기반으로 AI 맞춤 케어를 도와드릴게요!',
              style: const TextStyle(color: Colors.black87, fontSize: 13),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildSectionTitle(String title) {
    return Text(title, style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold, color: Colors.grey[800]));
  }

  InputDecoration _inputDecoration(String hint) {
    return InputDecoration(
      hintText: hint,
      hintStyle: TextStyle(color: Colors.grey[400]),
      filled: true,
      fillColor: Colors.white,
      contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 16),
      enabledBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(12),
        borderSide: BorderSide(color: Colors.grey[300]!),
      ),
      focusedBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(12),
        borderSide: const BorderSide(color: Colors.orange),
      ),
    );
  }

  Widget _buildCheckRadio(String label, String groupValue, Function(String) onTap) {
    bool isSelected = groupValue == label;
    return GestureDetector(
      onTap: () => onTap(label),
      child: Padding(
        padding: const EdgeInsets.symmetric(vertical: 4),
        child: Row(
          children: [
            Icon(
              isSelected ? Icons.check_circle : Icons.radio_button_unchecked,
              color: isSelected ? Colors.orange : Colors.grey[300],
              size: 24,
            ),
            const SizedBox(width: 8),
            Text(label, style: const TextStyle(fontSize: 15, color: Colors.black87)),
          ],
        ),
      ),
    );
  }

  Widget _buildCircleRadio(String label, String groupValue, Function(String) onTap) {
    bool isSelected = groupValue == label;
    return GestureDetector(
      onTap: () => onTap(label),
      child: Container(
        padding: const EdgeInsets.symmetric(vertical: 8),
        color: Colors.transparent,
        child: Row(
          children: [
            Container(
              width: 20,
              height: 20,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                border: Border.all(color: isSelected ? Colors.orange : Colors.grey[300]!, width: 2),
              ),
              child: isSelected
                  ? Center(child: Container(width: 10, height: 10, decoration: const BoxDecoration(color: Colors.orange, shape: BoxShape.circle)))
                  : null,
            ),
            const SizedBox(width: 10),
            Text(label, style: const TextStyle(fontSize: 15, color: Colors.black87)),
          ],
        ),
      ),
    );
  }

  Widget _buildSpeciesCard(String label, IconData icon) {
    bool isSelected = _selectedSpecies == label;
    return Expanded(
      child: GestureDetector(
        onTap: () => setState(() => _selectedSpecies = label),
        child: Container(
          padding: const EdgeInsets.symmetric(vertical: 24),
          decoration: BoxDecoration(
            color: Colors.white,
            borderRadius: BorderRadius.circular(12),
            border: Border.all(
              color: isSelected ? Colors.orange : Colors.grey[300]!,
              width: isSelected ? 2 : 1,
            ),
          ),
          child: Column(
            children: [
              Icon(icon, color: isSelected ? Colors.orange : Colors.grey[400], size: 32),
              const SizedBox(height: 8),
              Text(label, style: TextStyle(color: isSelected ? Colors.orange : Colors.grey[600], fontSize: 13)),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildGenderCard(String label, IconData icon) {
    bool isSelected = _selectedGender == label;
    return GestureDetector(
      onTap: () => setState(() => _selectedGender = label),
      child: Container(
        padding: const EdgeInsets.symmetric(vertical: 20),
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(12),
          border: Border.all(
            color: isSelected ? Colors.orange : Colors.grey[300]!,
            width: isSelected ? 2 : 1,
          ),
        ),
        child: Column(
          children: [
            Icon(icon, color: isSelected ? Colors.orange : Colors.grey[400], size: 40),
            const SizedBox(height: 8),
            Text(label, style: TextStyle(color: isSelected ? Colors.orange : Colors.grey[600])),
          ],
        ),
      ),
    );
  }
}