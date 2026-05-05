import 'package:flutter/material.dart';
import 'package:flutter/cupertino.dart';
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

  // 설정 상태 변수
  int _recordingInterval = 60; // 디폴트 60분
  String _diaryCoverType = 'happy'; // 디폴트: 행복 우선

  final String baseUrl = "http://localhost:8080";

  @override
  void initState() {
    super.initState();
    _petName = widget.petData?['pet_name'] ?? '우리';
    _profileImageUrl = widget.petData?['profile_image_url'];
    _coverImageUrl = widget.petData?['cover_image_url'];
    _loadSettings();
  }

  Future<void> _loadSettings() async {
    try {
      final url = Uri.parse('$baseUrl/api/settings/${widget.userId}');
      final response = await http.get(url);
      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        if (data['status'] == 'success' && data['settings'] != null) {
          setState(() {
            _recordingInterval = data['settings']['recording_interval'] ?? 60;
            _diaryCoverType = data['settings']['diary_cover_type'] ?? 'happy';
          });
        }
      }
    } catch (e) {
      // 설정 로드 실패 시 기본값 유지
      debugPrint('설정 로드 실패: $e');
    }
  }

  Future<void> _saveSettings() async {
    try {
      final url = Uri.parse('$baseUrl/api/settings/${widget.userId}');
      final response = await http.post(
        url,
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
          'recording_interval': _recordingInterval,
          'diary_cover_type': _diaryCoverType,
        }),
      );
      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        if (data['status'] == 'success') {
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(content: Text('설정이 저장되었습니다.')),
          );
        }
      }
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('설정 저장 실패: $e')),
      );
    }
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

                _buildSectionTitle('설정'),
                _buildSettingsTile(
                  Icons.videocam_outlined,
                  '촬영 주기 설정',
                  Colors.teal,
                  trailing: Text(
                    '$_recordingInterval분',
                    style: const TextStyle(color: Colors.orange, fontSize: 13, fontWeight: FontWeight.w600),
                  ),
                  onTap: () => _showRecordingIntervalPicker(),
                ),
                _buildSettingsTile(
                  Icons.photo_library_outlined,
                  '일기 대표 사진 설정',
                  Colors.deepPurple,
                  trailing: Text(
                    _diaryCoverType == 'happy' ? '행복 우선' : '감정 빈도 우선',
                    style: const TextStyle(color: Colors.orange, fontSize: 13, fontWeight: FontWeight.w600),
                  ),
                  onTap: () => _showDiaryCoverTypeDialog(),
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

  void _showRecordingIntervalPicker() {
    final List<int> intervals = [20, 30, 40, 50, 60];
    int tempSelected = _recordingInterval;
    int initialIndex = intervals.indexOf(_recordingInterval);
    if (initialIndex < 0) initialIndex = intervals.length - 1; // 기본 60분

    showModalBottomSheet(
      context: context,
      backgroundColor: Colors.transparent,
      builder: (context) => StatefulBuilder(
        builder: (context, setModalState) {
          return Container(
            decoration: const BoxDecoration(
              color: Color(0xFF2C2C2E),
              borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
            ),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                // 상단 핸들 바
                const SizedBox(height: 8),
                Container(
                  width: 40, height: 4,
                  decoration: BoxDecoration(
                    color: Colors.grey[600],
                    borderRadius: BorderRadius.circular(2),
                  ),
                ),
                const SizedBox(height: 16),
                const Text(
                  '촬영 주기 설정',
                  style: TextStyle(color: Colors.white, fontSize: 18, fontWeight: FontWeight.bold),
                ),
                const SizedBox(height: 4),
                Text(
                  '반려동물 촬영 간격을 설정하세요',
                  style: TextStyle(color: Colors.grey[400], fontSize: 13),
                ),
                const SizedBox(height: 16),
                // CupertinoPicker 휠
                SizedBox(
                  height: 200,
                  child: CupertinoPicker(
                    scrollController: FixedExtentScrollController(initialItem: initialIndex),
                    magnification: 1.3,
                    squeeze: 1.2,
                    useMagnifier: true,
                    itemExtent: 50,
                    selectionOverlay: Container(
                      decoration: BoxDecoration(
                        border: Border.symmetric(
                          horizontal: BorderSide(color: Colors.grey[600]!, width: 0.5),
                        ),
                        color: Colors.white.withOpacity(0.08),
                      ),
                    ),
                    onSelectedItemChanged: (index) {
                      setModalState(() {
                        tempSelected = intervals[index];
                      });
                    },
                    children: intervals.map((val) {
                      final isSelected = val == tempSelected;
                      return Center(
                        child: Text(
                          '$val분',
                          style: TextStyle(
                            color: isSelected ? Colors.white : Colors.grey[500],
                            fontSize: isSelected ? 24 : 18,
                            fontWeight: isSelected ? FontWeight.bold : FontWeight.normal,
                          ),
                        ),
                      );
                    }).toList(),
                  ),
                ),
                const SizedBox(height: 16),
                // 완료 버튼
                Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 24),
                  child: SizedBox(
                    width: double.infinity,
                    child: ElevatedButton(
                      onPressed: () {
                        setState(() {
                          _recordingInterval = tempSelected;
                        });
                        _saveSettings();
                        Navigator.pop(context);
                      },
                      style: ElevatedButton.styleFrom(
                        backgroundColor: Colors.orange,
                        foregroundColor: Colors.white,
                        elevation: 0,
                        padding: const EdgeInsets.symmetric(vertical: 14),
                        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                      ),
                      child: const Text('완료', style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold)),
                    ),
                  ),
                ),
                const SizedBox(height: 24),
              ],
            ),
          );
        },
      ),
    );
  }

  void _showDiaryCoverTypeDialog() {
    showModalBottomSheet(
      context: context,
      backgroundColor: Colors.transparent,
      builder: (context) {
        return Container(
          decoration: const BoxDecoration(
            color: Color(0xFF2C2C2E),
            borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
          ),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              const SizedBox(height: 8),
              Container(
                width: 40, height: 4,
                decoration: BoxDecoration(
                  color: Colors.grey[600],
                  borderRadius: BorderRadius.circular(2),
                ),
              ),
              const SizedBox(height: 16),
              const Text(
                '일기 대표 사진 설정',
                style: TextStyle(color: Colors.white, fontSize: 18, fontWeight: FontWeight.bold),
              ),
              const SizedBox(height: 4),
              Text(
                '일기에 표시될 대표 사진의 기준을 선택하세요',
                style: TextStyle(color: Colors.grey[400], fontSize: 13),
              ),
              const SizedBox(height: 24),
              // 행복 우선 버튼
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 24),
                child: SizedBox(
                  width: double.infinity,
                  child: ElevatedButton.icon(
                    onPressed: () {
                      setState(() => _diaryCoverType = 'happy');
                      _saveSettings();
                      Navigator.pop(context);
                    },
                    icon: const Icon(Icons.sentiment_very_satisfied, size: 22),
                    label: const Text('행복 우선', style: TextStyle(fontSize: 15, fontWeight: FontWeight.w600)),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: _diaryCoverType == 'happy' ? Colors.orange : Colors.grey[700],
                      foregroundColor: Colors.white,
                      elevation: 0,
                      padding: const EdgeInsets.symmetric(vertical: 14),
                      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                    ),
                  ),
                ),
              ),
              const SizedBox(height: 12),
              // 감정 빈도 우선 버튼
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 24),
                child: SizedBox(
                  width: double.infinity,
                  child: ElevatedButton.icon(
                    onPressed: () {
                      setState(() => _diaryCoverType = 'frequent');
                      _saveSettings();
                      Navigator.pop(context);
                    },
                    icon: const Icon(Icons.bar_chart_rounded, size: 22),
                    label: const Text('감정 빈도 우선', style: TextStyle(fontSize: 15, fontWeight: FontWeight.w600)),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: _diaryCoverType == 'frequent' ? Colors.orange : Colors.grey[700],
                      foregroundColor: Colors.white,
                      elevation: 0,
                      padding: const EdgeInsets.symmetric(vertical: 14),
                      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                    ),
                  ),
                ),
              ),
              const SizedBox(height: 24),
            ],
          ),
        );
      },
    );
  }

  Widget _buildSettingsTile(IconData icon, String title, Color color, {Widget? trailing, VoidCallback? onTap}) {
    return ListTile(
      leading: Container(
        padding: const EdgeInsets.all(8),
        decoration: BoxDecoration(color: color.withOpacity(0.1), shape: BoxShape.circle),
        child: Icon(icon, color: color),
      ),
      title: Text(title, style: const TextStyle(fontWeight: FontWeight.w600)),
      trailing: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          if (trailing != null) trailing,
          const SizedBox(width: 8),
          const Icon(Icons.arrow_forward_ios, size: 14, color: Colors.grey),
        ],
      ),
      onTap: onTap,
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