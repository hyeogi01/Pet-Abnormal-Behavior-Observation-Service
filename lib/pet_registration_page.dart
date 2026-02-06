import 'package:flutter/material.dart';

class PetRegistrationPage extends StatefulWidget {
  final String petName;

  const PetRegistrationPage({Key? key, required this.petName}) : super(key: key);

  @override
  _PetRegistrationPageState createState() => _PetRegistrationPageState();
}

class _PetRegistrationPageState extends State<PetRegistrationPage> {
  // 기존 변수들
  String _selectedSpecies = '강아지';
  String _selectedGender = '남아';
  TextEditingController _speciesDetailController = TextEditingController();
  DateTime? _birthDate;

  // 새로 추가된 상태 변수들
  String _neuteredStatus = '안했어요'; // 중성화 기본값
  TextEditingController _diseaseController = TextEditingController(); // 질환 입력
  String _separationAnxiety = '모르겠어요'; // 분리불안 기본값

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.white,
      appBar: AppBar(
        backgroundColor: Colors.white,
        elevation: 0,
        leading: IconButton(
          icon: Icon(Icons.arrow_back_ios, color: Colors.black, size: 20),
          onPressed: () => Navigator.pop(context),
        ),
        title: Text('반려동물 등록', style: TextStyle(color: Colors.black, fontSize: 16, fontWeight: FontWeight.bold)),
        centerTitle: true,
        bottom: PreferredSize(
          preferredSize: Size.fromHeight(4.0),
          child: LinearProgressIndicator(
            value: 1.0,
            backgroundColor: Colors.grey[200],
            valueColor: AlwaysStoppedAnimation<Color>(Colors.orange),
            minHeight: 2,
          ),
        ),
      ),
      body: SingleChildScrollView(
        child: Padding(
          padding: EdgeInsets.symmetric(horizontal: 24, vertical: 32),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // --- 상단 헤더 ---
              Text('궁금해요', style: TextStyle(fontSize: 16, color: Colors.black54)),
              SizedBox(height: 8),
              RichText(
                text: TextSpan(
                  children: [
                    TextSpan(
                      text: '${widget.petName}에 대해\n더 알려 주실래요? 🐶',
                      style: TextStyle(fontSize: 26, fontWeight: FontWeight.bold, color: Colors.black, height: 1.3),
                    ),
                  ],
                ),
              ),
              SizedBox(height: 24),
              Container(
                width: double.infinity,
                padding: EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: Colors.grey[100],
                  borderRadius: BorderRadius.circular(12),
                ),
                child: Row(
                  children: [
                    Icon(Icons.pets, color: Colors.orange, size: 20),
                    SizedBox(width: 10),
                    Expanded(
                      child: Text(
                        '${widget.petName} 정보를 기반으로 AI 맞춤 케어를 도와드릴게요!',
                        style: TextStyle(color: Colors.black87, fontSize: 13),
                      ),
                    ),
                  ],
                ),
              ),
              SizedBox(height: 40),

              // --- 1. 종 선택 ---
              _buildSectionTitle('종'),
              SizedBox(height: 12),
              Row(
                children: [
                  _buildSpeciesCard('강아지', Icons.pets),
                  SizedBox(width: 12),
                  _buildSpeciesCard('고양이', Icons.catching_pokemon),
                  SizedBox(width: 12),
                  _buildSpeciesCard('다른 동물', Icons.emoji_nature),
                ],
              ),
              SizedBox(height: 12),
              TextField(
                controller: _speciesDetailController,
                decoration: _inputDecoration('품종을 입력해주세요 (예: 말티즈)'),
              ),
              SizedBox(height: 40),

              // --- 2. 성별 선택 ---
              _buildSectionTitle('성별'),
              SizedBox(height: 12),
              Row(
                children: [
                  Expanded(child: _buildGenderCard('남아', Icons.male)),
                  SizedBox(width: 12),
                  Expanded(child: _buildGenderCard('여아', Icons.female)),
                ],
              ),
              SizedBox(height: 40),

              // --- 3. 생년월일 ---
              _buildSectionTitle('생년월일'),
              SizedBox(height: 12),
              GestureDetector(
                onTap: () async {
                  final DateTime? picked = await showDatePicker(
                    context: context,
                    initialDate: DateTime.now(),
                    firstDate: DateTime(2000),
                    lastDate: DateTime.now(),
                    builder: (context, child) {
                      return Theme(
                        data: Theme.of(context).copyWith(
                          colorScheme: ColorScheme.light(primary: Colors.orange),
                        ),
                        child: child!,
                      );
                    },
                  );
                  if (picked != null) {
                    setState(() {
                      _birthDate = picked;
                    });
                  }
                },
                child: Container(
                  width: double.infinity,
                  padding: EdgeInsets.symmetric(vertical: 16, horizontal: 16),
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
              SizedBox(height: 40),

              // --- 4. 중성화 여부 (NEW) ---
              _buildSectionTitle('중성화 여부'),
              SizedBox(height: 12),
              Column(
                children: [
                  _buildCheckRadio('안했어요', _neuteredStatus, (val) => setState(() => _neuteredStatus = val)),
                  SizedBox(height: 10),
                  _buildCheckRadio('중성화했어요', _neuteredStatus, (val) => setState(() => _neuteredStatus = val)),
                ],
              ),
              SizedBox(height: 40),

              // --- 5. 앓고 있는 질환 (NEW) ---
              _buildSectionTitle('앓고 있는 질환'),
              SizedBox(height: 12),
              TextField(
                controller: _diseaseController,
                decoration: _inputDecoration('앓고 있는 질환 정보를 알려주세요'),
              ),
              SizedBox(height: 40),

              // --- 6. 분리불안 여부 (NEW) ---
              _buildSectionTitle('분리불안 여부'),
              SizedBox(height: 12),
              Column(
                children: [
                  _buildCircleRadio('있어요', _separationAnxiety, (val) => setState(() => _separationAnxiety = val)),
                  _buildCircleRadio('없어요', _separationAnxiety, (val) => setState(() => _separationAnxiety = val)),
                  _buildCircleRadio('모르겠어요', _separationAnxiety, (val) => setState(() => _separationAnxiety = val)),
                ],
              ),
              SizedBox(height: 50),

              // --- 완료 버튼 ---
              SizedBox(
                width: double.infinity,
                height: 56,
                child: ElevatedButton(
                  onPressed: () {
                    // 데이터 출력 확인용
                    print('이름: ${widget.petName}');
                    print('종: $_selectedSpecies, 성별: $_selectedGender');
                    print('생일: $_birthDate');
                    print('중성화: $_neuteredStatus');
                    print('질환: ${_diseaseController.text}');
                    print('분리불안: $_separationAnxiety');

                    ScaffoldMessenger.of(context).showSnackBar(
                      SnackBar(content: Text('${widget.petName} 등록이 완료되었습니다!')),
                    );
                  },
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.grey[300], // 모든 필수 입력 완료 시 주황색으로 변경하는 로직 추가 가능
                    elevation: 0,
                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                  ),
                  child: Text('다음으로', style: TextStyle(color: Colors.white, fontSize: 16, fontWeight: FontWeight.bold)),
                ),
              ),
              SizedBox(height: 20),
            ],
          ),
        ),
      ),
    );
  }

  // --- 위젯 헬퍼 함수들 ---

  Widget _buildSectionTitle(String title) {
    return Text(
      title,
      style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold, color: Colors.grey[800]),
    );
  }

  InputDecoration _inputDecoration(String hint) {
    return InputDecoration(
      hintText: hint,
      hintStyle: TextStyle(color: Colors.grey[400]),
      filled: true,
      fillColor: Colors.white,
      contentPadding: EdgeInsets.symmetric(horizontal: 16, vertical: 16),
      enabledBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(12),
        borderSide: BorderSide(color: Colors.grey[300]!),
      ),
      focusedBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(12),
        borderSide: BorderSide(color: Colors.orange),
      ),
    );
  }

  // 중성화 여부용 체크형 라디오 버튼 (이미지 참고: 주황색 체크 아이콘)
  Widget _buildCheckRadio(String label, String groupValue, Function(String) onTap) {
    bool isSelected = groupValue == label;
    return GestureDetector(
      onTap: () => onTap(label),
      child: Row(
        children: [
          Icon(
            isSelected ? Icons.check_circle : Icons.radio_button_unchecked,
            color: isSelected ? Colors.orange : Colors.grey[300],
            size: 24,
          ),
          SizedBox(width: 8),
          Text(label, style: TextStyle(fontSize: 15, color: Colors.black87)),
        ],
      ),
    );
  }

  // 분리불안용 원형 라디오 버튼
  Widget _buildCircleRadio(String label, String groupValue, Function(String) onTap) {
    bool isSelected = groupValue == label;
    return GestureDetector(
      onTap: () => onTap(label),
      child: Container(
        padding: EdgeInsets.symmetric(vertical: 8),
        color: Colors.transparent, // 터치 영역 확보
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
                  ? Center(child: Container(width: 10, height: 10, decoration: BoxDecoration(color: Colors.orange, shape: BoxShape.circle)))
                  : null,
            ),
            SizedBox(width: 10),
            Text(label, style: TextStyle(fontSize: 15, color: Colors.black87)),
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
          padding: EdgeInsets.symmetric(vertical: 24),
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
              SizedBox(height: 8),
              Text(
                label,
                style: TextStyle(
                  color: isSelected ? Colors.orange : Colors.grey[600],
                  fontWeight: isSelected ? FontWeight.bold : FontWeight.normal,
                  fontSize: 13,
                ),
              ),
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
        padding: EdgeInsets.symmetric(vertical: 20),
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
            SizedBox(height: 8),
            Text(
              label,
              style: TextStyle(
                color: isSelected ? Colors.orange : Colors.grey[600],
                fontWeight: isSelected ? FontWeight.bold : FontWeight.normal,
              ),
            ),
          ],
        ),
      ),
    );
  }
}