import 'package:flutter/material.dart';
import 'pet_registration_page.dart'; // 다음 페이지 import
// import 'package:http/http.dart' as http; // 상단에 추가
// import 'dart:convert'; // JSON 변환을 위해 추가

class PetNameInputPage extends StatefulWidget {
  final String userId; // [추가] 부모로부터 전달받을 ID

  const PetNameInputPage({Key? key, required this.userId}) : super(key: key); // [수정]

  @override
  _PetNameInputPageState createState() => _PetNameInputPageState();
}

class _PetNameInputPageState extends State<PetNameInputPage> {
  final TextEditingController _nameController = TextEditingController();
  bool _isButtonEnabled = false;
  // 데이터를 서버로 보내는 비동기 함수
  Future<void> _savePetName(String name) async {
    // final Uri url = Uri.parse('http://localhost:8000/user-input/');
    //
    // try {
    //   final response = await http.post(
    //     url,
    //     headers: {'Content-Type': 'application/json'},
    //     body: jsonEncode({
    //       'user_id': name,
    //       'message': '신규 등록', // 초기값
    //       'priority': 1,
    //     }),
    //   );
    //
    //   if (response.statusCode == 200) {
    //     print('서버 저장 성공!');
    //   } else {
    //     print('서버 오류: ${response.statusCode}');
    //   }
    // } catch (e) {
    //   print('네트워크 에러: $e');
    // }
  }

  @override
  void initState() {
    super.initState();
    _nameController.addListener(() {
      setState(() {
        _isButtonEnabled = _nameController.text.trim().isNotEmpty;
      });
    });
  }

  @override
  void dispose() {
    _nameController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Colors.white,
        elevation: 0,
        leading: IconButton(
          icon: Icon(Icons.arrow_back_ios, color: Colors.black, size: 20),
          onPressed: () {},
        ),
        bottom: PreferredSize(
          preferredSize: Size.fromHeight(4.0),
          child: LinearProgressIndicator(
            value: 0.5,
            backgroundColor: Colors.grey[200],
            valueColor: AlwaysStoppedAnimation<Color>(Colors.orange),
            minHeight: 2,
          ),
        ),
      ),
      body: Padding(
        padding: EdgeInsets.symmetric(horizontal: 24, vertical: 32),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('반가워요!', style: TextStyle(fontSize: 16, color: Colors.black54)),
            SizedBox(height: 8),
            Text(
              '반려 동물의 이름은\n무엇인가요?',
              style: TextStyle(fontSize: 26, fontWeight: FontWeight.bold, color: Colors.black, height: 1.3),
            ),
            SizedBox(height: 40),
            TextField(
              controller: _nameController,
              autofocus: true,
              style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
              decoration: InputDecoration(
                hintText: '이름을 입력해주세요',
                hintStyle: TextStyle(color: Colors.grey[400], fontSize: 20),
                enabledBorder: UnderlineInputBorder(
                  borderSide: BorderSide(color: Colors.grey[300]!),
                ),
                focusedBorder: UnderlineInputBorder(
                  borderSide: BorderSide(color: Colors.orange, width: 2),
                ),
              ),
            ),
            Spacer(),
            SizedBox(
              width: double.infinity,
              height: 56,
              child: ElevatedButton(
                onPressed: _isButtonEnabled
                    ? () async {
                  // // 1. 서버로 이름 전송
                  // await _savePetName(_nameController.text);
                  Navigator.push(
                    context,
                    MaterialPageRoute(
                      builder: (context) => PetRegistrationPage(
                        petName: _nameController.text,
                        userId: widget.userId, // [수정] 임시 값 대신 받은 ID 전달
                      ),
                    ),
                  );
                }
                    : null,
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.orange,
                  disabledBackgroundColor: Colors.grey[300],
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
    );
  }
}