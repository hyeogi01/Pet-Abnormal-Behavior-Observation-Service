import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'dart:typed_data';

class ProfileEditPage extends StatefulWidget {
  final String userId;
  final String currentName;
  final String? currentImageUrl;

  const ProfileEditPage({super.key, required this.userId, required this.currentName, this.currentImageUrl});

  @override
  State<ProfileEditPage> createState() => _ProfileEditPageState();
}

class _ProfileEditPageState extends State<ProfileEditPage> {
  late TextEditingController _nameController;
  
  Uint8List? _selectedImageBytes;
  String? _selectedImageName;
  
  String? _currentImageUrl;
  bool _isLoading = false;

  final String baseUrl = "http://localhost:8080";

  @override
  void initState() {
    super.initState();
    _nameController = TextEditingController(text: widget.currentName);
    _currentImageUrl = widget.currentImageUrl;
  }

  @override
  void dispose() {
    _nameController.dispose();
    super.dispose();
  }

  Future<void> _pickImage() async {
    final picker = ImagePicker();
    final pickedFile = await picker.pickImage(source: ImageSource.gallery);
    if (pickedFile != null) {
      final bytes = await pickedFile.readAsBytes();
      setState(() {
        _selectedImageBytes = bytes;
        _selectedImageName = pickedFile.name;
      });
    }
  }

  Future<void> _saveProfile() async {
    setState(() => _isLoading = true);

    String? newImageUrl = _currentImageUrl;

    // 1. 이미지 변경사항이 있다면 업로드 (바이트 배열 기반, 웹 호환)
    if (_selectedImageBytes != null && _selectedImageName != null) {
      final uri = Uri.parse('$baseUrl/api/profile-image/${widget.userId}');
      var request = http.MultipartRequest('POST', uri);
      request.fields['is_cover'] = 'false';
      
      var multipartFile = http.MultipartFile.fromBytes('file', _selectedImageBytes!, filename: _selectedImageName);
      request.files.add(multipartFile);

      try {
        var streamedResponse = await request.send();
        var response = await http.Response.fromStream(streamedResponse);
        if (response.statusCode == 200) {
          var data = jsonDecode(response.body);
          if (data['status'] == 'success') {
            newImageUrl = data['image_url'];
          }
        }
      } catch (e) {
        print("Image upload failed: $e");
      }
    }

    // 2. 이름 갱신 (항상 진행)
    try {
      final nameUri = Uri.parse('$baseUrl/api/update-pet-info/${widget.userId}');
      final response = await http.post(
        nameUri,
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({"pet_name": _nameController.text})
      );
      
      if (response.statusCode == 200) {
        // 성공 시 이전 페이지로 갱신된 데이터 전달
        Navigator.pop(context, {
          'name': _nameController.text,
          'image_url': newImageUrl,
        });
      } else {
        ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('이름 변경 실패')));
      }
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('통신 오류')));
    } finally {
      setState(() => _isLoading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.white,
      appBar: AppBar(
        title: const Text('프로필 수정', style: TextStyle(color: Colors.black, fontWeight: FontWeight.bold)),
        backgroundColor: Colors.white,
        iconTheme: const IconThemeData(color: Colors.black),
        elevation: 0,
        actions: [
          _isLoading 
            ? const Padding(
                padding: EdgeInsets.all(12),
                child: SizedBox(width: 20, height: 20, child: CircularProgressIndicator(strokeWidth: 2)),
              )
            : TextButton(
                onPressed: _saveProfile,
                child: const Text('저장', style: TextStyle(color: Colors.blueAccent, fontWeight: FontWeight.bold, fontSize: 16)),
              )
        ],
      ),
      body: Padding(
        padding: const EdgeInsets.all(24.0),
        child: Column(
          children: [
            const SizedBox(height: 20),
            Center(
              child: Stack(
                children: [
                  Container(
                    decoration: BoxDecoration(
                      shape: BoxShape.circle,
                      boxShadow: const [BoxShadow(color: Colors.black12, blurRadius: 10, offset: Offset(0, 5))],
                    ),
                    child: CircleAvatar(
                      radius: 60,
                      backgroundColor: Colors.grey[200],
                      backgroundImage: _selectedImageBytes != null
                          ? MemoryImage(_selectedImageBytes!) as ImageProvider
                          : (_currentImageUrl != null && _currentImageUrl!.isNotEmpty
                              ? NetworkImage(_currentImageUrl!)
                              : null),
                      child: (_selectedImageBytes == null && (_currentImageUrl == null || _currentImageUrl!.isEmpty))
                          ? const Icon(Icons.pets, size: 50, color: Colors.white)
                          : null,
                    ),
                  ),
                  Positioned(
                    bottom: 0,
                    right: 0,
                    child: GestureDetector(
                      onTap: _pickImage,
                      child: const CircleAvatar(
                        radius: 20,
                        backgroundColor: Colors.blueAccent,
                        child: Icon(Icons.camera_alt, color: Colors.white, size: 20),
                      ),
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 40),
            TextField(
              controller: _nameController,
              decoration: InputDecoration(
                labelText: '반려동물 이름',
                border: OutlineInputBorder(borderRadius: BorderRadius.circular(12), borderSide: BorderSide.none),
                filled: true,
                fillColor: Colors.grey[100],
                prefixIcon: const Icon(Icons.pets, color: Colors.blueGrey),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
