import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'dart:io';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'package:pet_diary/discription/onboarding_page.dart'; // import for OnboardingPage
import 'profile_edit_page.dart';
import 'weight_history_page.dart';
import 'account_settings_page.dart';

class MyPage extends StatefulWidget {
  final Map<String, dynamic>? petData;
  final String userId;

  const MyPage({super.key, this.petData, required this.userId});

  @override
  State<MyPage> createState() => _MyPageState();
}

class _MyPageState extends State<MyPage> {
  bool _isPushEnabled = true;
  String _petName = '우리';
  String? _profileImageUrl;
  String? _coverImageUrl;

  final String baseUrl = "http://localhost:8080";

  @override
  void initState() {
    super.initState();
    _petName = widget.petData?['pet_name'] ?? '우리';
    _profileImageUrl = widget.petData?['profile_image_url'];
    _coverImageUrl = widget.petData?['cover_image_url'];
  }

  Future<void> _pickImage(bool isCover) async {
    final picker = ImagePicker();
    final pickedFile = await picker.pickImage(source: ImageSource.gallery);
    if (pickedFile != null) {
      final bytes = await pickedFile.readAsBytes();
      await _uploadImage(bytes, pickedFile.name, isCover);
    }
  }

  Future<void> _uploadImage(List<int> bytes, String filename, bool isCover) async {
    final uri = Uri.parse('$baseUrl/api/profile-image/${widget.userId}');
    var request = http.MultipartRequest('POST', uri);
    request.fields['is_cover'] = isCover ? 'true' : 'false';
    
    var multipartFile = http.MultipartFile.fromBytes('file', bytes, filename: filename);
    request.files.add(multipartFile);

    try {
      var streamedResponse = await request.send();
      var response = await http.Response.fromStream(streamedResponse);
      if (response.statusCode == 200) {
        var data = jsonDecode(response.body);
        if (data['status'] == 'success') {
          setState(() {
            if (isCover) {
              _coverImageUrl = data['image_url'];
            } else {
              _profileImageUrl = data['image_url'];
            }
          });
          ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('사진이 업데이트되었습니다.')));
        } else {
          ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('업로드 실패: ${data['message']}')));
        }
      }
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('서버 통신 에러: $e')));
    }
  }

  @override
  Widget build(BuildContext context) {
    return SingleChildScrollView(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _buildProfileHeader(),
          const SizedBox(height: 10),

          Padding(
            padding: const EdgeInsets.all(16.0),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                _buildSectionTitle('신체 정보 기록'),
                GestureDetector(
                  onTap: () {
                    Navigator.push(context, MaterialPageRoute(builder: (context) => WeightHistoryPage(userId: widget.userId)));
                  },
                  child: _buildSimpleCard(Icons.monitor_weight_outlined, '몸무게 기록 및 차트', '상세보기', Colors.blue, showArrow: true),
                ),
                const SizedBox(height: 20),
                
                _buildSectionTitle('알림 설정 (Detection System)'),
                SwitchListTile(
                  secondary: Container(
                    padding: const EdgeInsets.all(8),
                    decoration: BoxDecoration(color: Colors.orange.withOpacity(0.1), shape: BoxShape.circle),
                    child: const Icon(Icons.notifications_active_outlined, color: Colors.orange),
                  ),
                  title: const Text('이상행동 푸시 알림', style: TextStyle(fontWeight: FontWeight.w500)),
                  value: _isPushEnabled,
                  activeColor: Colors.orange,
                  onChanged: (val) => setState(() => _isPushEnabled = val),
                  shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(15)),
                ),
                const SizedBox(height: 20),
                
                _buildSectionTitle('보안 및 계정'),
                _buildListTile(Icons.manage_accounts_outlined, '계정 관리', Colors.blueGrey, onTap: () {
                  Navigator.push(context, MaterialPageRoute(builder: (context) => AccountSettingsPage(userId: widget.userId)));
                }),
                _buildListTile(Icons.logout, '로그아웃', Colors.redAccent, onTap: () => _showLogoutDialog()),
                const SizedBox(height: 20),
                
                _buildSectionTitle('앱 정보'),
                const ListTile(
                  leading: Icon(Icons.info_outline),
                  title: Text('앱 버전'),
                  trailing: Text('v1.0.1', style: TextStyle(color: Colors.grey)),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  void _showLogoutDialog() {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('로그아웃'),
        content: const Text('정말 로그아웃 하시겠습니까?'),
        actions: [
          TextButton(onPressed: () => Navigator.pop(context), child: const Text('취소')),
          TextButton(
            onPressed: () {
              Navigator.pop(context); // close dialog
              Navigator.pushAndRemoveUntil(
                context, 
                MaterialPageRoute(builder: (context) => const OnboardingPage()), 
                (route) => false
              );
            },
            child: const Text('로그아웃', style: TextStyle(color: Colors.red)),
          ),
        ],
      )
    );
  }

  Widget _buildProfileHeader() {
    return Stack(
      alignment: Alignment.center,
      children: [
        Column(
          children: [
            Container(
              height: 210,
              width: double.infinity,
              decoration: BoxDecoration(
                color: Colors.grey[300],
                image: _coverImageUrl != null && _coverImageUrl!.isNotEmpty
                    ? DecorationImage(image: NetworkImage(_coverImageUrl!), fit: BoxFit.cover)
                    : null,
              ),
              child: Stack(
                children: [
                  if (_coverImageUrl == null || _coverImageUrl!.isEmpty)
                    const Center(child: Icon(Icons.photo, size: 50, color: Colors.black26)),
                  Align(
                    alignment: Alignment.topRight,
                    child: IconButton(
                      padding: const EdgeInsets.only(top: 5, right: 10),
                      icon: const CircleAvatar(
                        backgroundColor: Colors.black45,
                        child: Icon(Icons.camera_alt, color: Colors.white, size: 18),
                      ),
                      onPressed: () => _pickImage(true),
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 50),
          ],
        ),
        Positioned(
          top: 100,
          child: Stack(
            children: [
              Container(
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  border: Border.all(color: Colors.white, width: 4),
                  boxShadow: const [BoxShadow(color: Colors.black26, blurRadius: 8, offset: Offset(0, 4))],
                ),
                child: CircleAvatar(
                  radius: 50,
                  backgroundColor: Colors.blueGrey[100],
                  backgroundImage: _profileImageUrl != null && _profileImageUrl!.isNotEmpty
                      ? NetworkImage(_profileImageUrl!)
                      : null,
                  child: (_profileImageUrl == null || _profileImageUrl!.isEmpty)
                      ? const Icon(Icons.pets, size: 40, color: Colors.white)
                      : null,
                ),
              ),
              Positioned(
                bottom: 0,
                right: 0,
                child: GestureDetector(
                  onTap: () async {
                    final updatedData = await Navigator.push(
                      context,
                      MaterialPageRoute(
                        builder: (context) => ProfileEditPage(
                          userId: widget.userId,
                          currentName: _petName,
                          currentImageUrl: _profileImageUrl,
                        )
                      ),
                    );
                    if (updatedData != null) {
                      setState(() {
                        if (updatedData['name'] != null) _petName = updatedData['name'];
                        if (updatedData['image_url'] != null) _profileImageUrl = updatedData['image_url'];
                      });
                    }
                  },
                  child: const CircleAvatar(
                    radius: 18,
                    backgroundColor: Colors.blueAccent,
                    child: Icon(Icons.edit, color: Colors.white, size: 18),
                  ),
                ),
              ),
            ],
          ),
        ),
        Positioned(
          bottom: 0,
          child: Column(
            children: [
              Text('$_petName네', style: const TextStyle(fontSize: 20, fontWeight: FontWeight.bold)),
              Text(widget.userId, style: TextStyle(color: Colors.grey[600], fontSize: 14)), 
            ],
          ),
        ),
      ],
    );
  }

  Widget _buildSectionTitle(String title) => Padding(
    padding: const EdgeInsets.only(left: 8, bottom: 8),
    child: Text(title, style: const TextStyle(fontSize: 14, fontWeight: FontWeight.bold, color: Colors.blueGrey)),
  );

  Widget _buildSimpleCard(IconData icon, String title, String trailing, Color color, {bool showArrow = false}) => Card(
    elevation: 0, color: Colors.blueGrey[50],
    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(15)),
    child: ListTile(
      leading: Container(
        padding: const EdgeInsets.all(8),
        decoration: BoxDecoration(color: color.withOpacity(0.1), shape: BoxShape.circle),
        child: Icon(icon, color: color),
      ),
      title: Text(title, style: const TextStyle(fontWeight: FontWeight.w600)),
      trailing: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Text(trailing, style: TextStyle(fontWeight: FontWeight.bold, color: Colors.grey[700])),
          if (showArrow) ...[
            const SizedBox(width: 8),
            const Icon(Icons.arrow_forward_ios, size: 14, color: Colors.grey),
          ]
        ],
      )
    ),
  );

  Widget _buildListTile(IconData icon, String title, Color color, {VoidCallback? onTap}) => ListTile(
    leading: Container(
        padding: const EdgeInsets.all(8),
        decoration: BoxDecoration(color: color.withOpacity(0.1), shape: BoxShape.circle),
        child: Icon(icon, color: color),
    ),
    title: Text(title, style: TextStyle(color: title == '로그아웃' ? Colors.redAccent : Colors.black87, fontWeight: title == '로그아웃' ? FontWeight.bold : FontWeight.w600)),
    trailing: title != '로그아웃' ? const Icon(Icons.arrow_forward_ios, size: 14, color: Colors.grey) : null,
    onTap: onTap ?? () {},
  );
}