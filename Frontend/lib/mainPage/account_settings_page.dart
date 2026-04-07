import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'package:pet_diary/discription/onboarding_page.dart'; // import 추가

class AccountSettingsPage extends StatefulWidget {
  final String userId;
  const AccountSettingsPage({super.key, required this.userId});

  @override
  State<AccountSettingsPage> createState() => _AccountSettingsPageState();
}

class _AccountSettingsPageState extends State<AccountSettingsPage> {
  final String baseUrl = "http://localhost:8080";

  void _showChangePasswordBottomSheet(BuildContext context) {
    final TextEditingController currentPwController = TextEditingController();
    final TextEditingController newPwController = TextEditingController();
    final TextEditingController confirmPwController = TextEditingController();
    bool isLoading = false;

    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.white,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(25)),
      ),
      builder: (context) => StatefulBuilder(
        builder: (context, setModalState) {
          return Padding(
            padding: EdgeInsets.only(
              bottom: MediaQuery.of(context).viewInsets.bottom,
              left: 24, right: 24, top: 16
            ),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                Center(
                  child: Container(
                    width: 50, height: 5,
                    decoration: BoxDecoration(color: Colors.grey[300], borderRadius: BorderRadius.circular(10)),
                  ),
                ),
                const SizedBox(height: 24),
                const Text('비밀번호 변경', style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold)),
                const SizedBox(height: 24),
                TextField(
                  controller: currentPwController,
                  obscureText: true, 
                  decoration: InputDecoration(
                    labelText: '현재 비밀번호', 
                    border: OutlineInputBorder(borderRadius: BorderRadius.circular(12)),
                    filled: true, fillColor: Colors.grey[50]
                  )
                ),
                const SizedBox(height: 16),
                TextField(
                  controller: newPwController,
                  obscureText: true, 
                  decoration: InputDecoration(
                    labelText: '새 비밀번호', 
                    border: OutlineInputBorder(borderRadius: BorderRadius.circular(12)),
                    filled: true, fillColor: Colors.grey[50]
                  )
                ),
                const SizedBox(height: 16),
                TextField(
                  controller: confirmPwController,
                  obscureText: true, 
                  decoration: InputDecoration(
                    labelText: '새 비밀번호 확인', 
                    border: OutlineInputBorder(borderRadius: BorderRadius.circular(12)),
                    filled: true, fillColor: Colors.grey[50]
                  )
                ),
                const SizedBox(height: 24),
                ElevatedButton(
                  onPressed: isLoading ? null : () async {
                    if (newPwController.text != confirmPwController.text) {
                      ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('새 비밀번호가 일치하지 않습니다.')));
                      return;
                    }
                    
                    setModalState(() => isLoading = true);
                    try {
                      final url = Uri.parse('$baseUrl/api/change-password/');
                      final response = await http.post(
                        url,
                        headers: {'Content-Type': 'application/json'},
                        body: jsonEncode({
                          "user_id": widget.userId,
                          "current_password": currentPwController.text,
                          "new_password": newPwController.text
                        }),
                      );
                      
                      final result = jsonDecode(response.body);
                      if (response.statusCode == 200 && result['status'] == 'success') {
                        Navigator.pop(context);
                        ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('비밀번호가 성공적으로 변경되었습니다.')));
                      } else {
                         ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('실패: ${result["message"]}')));
                      }
                    } catch (e) {
                      ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('비밀번호 변경 중 통신 오류가 발생했습니다.')));
                    } finally {
                      setModalState(() => isLoading = false);
                    }
                  },
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.blueAccent,
                    elevation: 0,
                    padding: const EdgeInsets.symmetric(vertical: 16),
                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                  ),
                  child: isLoading 
                    ? const SizedBox(width: 20, height: 20, child: CircularProgressIndicator(color: Colors.white, strokeWidth: 2))
                    : const Text('변경하기', style: TextStyle(fontSize: 16, color: Colors.white, fontWeight: FontWeight.bold)),
                ),
                const SizedBox(height: 32),
              ],
            ),
          );
        }
      )
    );
  }

  void _showDeleteAccountDialog(BuildContext context) {
    bool isLoading = false;
    showDialog(
      context: context,
      barrierDismissible: false,
      builder: (context) => StatefulBuilder(
        builder: (context, setDialogState) {
          return AlertDialog(
            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
            title: const Row(
              children: [
                Icon(Icons.warning_amber_rounded, color: Colors.redAccent, size: 28),
                SizedBox(width: 10),
                Text('계정 탈퇴', style: TextStyle(color: Colors.redAccent, fontWeight: FontWeight.bold)),
              ],
            ),
            content: const Text('정말 탈퇴하시겠습니까?\n모든 데이터가 삭제되며 복구할 수 없습니다.', style: TextStyle(height: 1.5, fontSize: 15)),
            actionsPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
            actions: [
              if (!isLoading)
                TextButton(
                  onPressed: () => Navigator.pop(context), 
                  child: const Text('취소', style: TextStyle(color: Colors.grey, fontSize: 16, fontWeight: FontWeight.w600))
                ),
              ElevatedButton(
                onPressed: isLoading ? null : () async {
                  setDialogState(() => isLoading = true);
                  try {
                    final url = Uri.parse('$baseUrl/api/delete-account/');
                    final response = await http.post(
                      url,
                      headers: {'Content-Type': 'application/json'},
                      body: jsonEncode({"user_id": widget.userId}),
                    );
                    
                    final result = jsonDecode(response.body);
                    if (response.statusCode == 200 && result['status'] == 'success') {
                      Navigator.pushAndRemoveUntil(
                          context,
                          MaterialPageRoute(builder: (context) => const OnboardingPage()),
                          (route) => false
                      );
                    } else {
                      Navigator.pop(context);
                      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('탈퇴 실패: ${result["message"]}')));
                    }
                  } catch (e) {
                     Navigator.pop(context);
                     ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('통신 오류가 발생했습니다.')));
                  }
                },
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.redAccent,
                  elevation: 0,
                  shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
                ),
                child: isLoading 
                  ? const SizedBox(width: 16, height: 16, child: CircularProgressIndicator(color: Colors.white, strokeWidth: 2))
                  : const Text('탈퇴하기', style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold)),
              ),
            ],
          );
        }
      )
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.white,
      appBar: AppBar(
        title: const Text('계정 관리', style: TextStyle(color: Colors.black, fontWeight: FontWeight.bold)),
        backgroundColor: Colors.white,
        iconTheme: const IconThemeData(color: Colors.black),
        elevation: 0,
        centerTitle: true,
      ),
      body: ListView(
        children: [
          const SizedBox(height: 10),
          ListTile(
            contentPadding: const EdgeInsets.symmetric(horizontal: 24, vertical: 8),
            leading: Container(
              padding: const EdgeInsets.all(10),
              decoration: BoxDecoration(color: Colors.blue.withOpacity(0.1), shape: BoxShape.circle),
              child: const Icon(Icons.lock_outline, color: Colors.blueAccent),
            ),
            title: const Text('비밀번호 변경', style: TextStyle(fontWeight: FontWeight.w600, fontSize: 16)),
            trailing: const Icon(Icons.arrow_forward_ios, size: 16, color: Colors.grey),
            onTap: () => _showChangePasswordBottomSheet(context),
          ),
          const Padding(padding: EdgeInsets.symmetric(horizontal: 24), child: Divider(height: 1, color: Colors.black12)),
          ListTile(
            contentPadding: const EdgeInsets.symmetric(horizontal: 24, vertical: 8),
            leading: Container(
              padding: const EdgeInsets.all(10),
              decoration: BoxDecoration(color: Colors.red.withOpacity(0.1), shape: BoxShape.circle),
              child: const Icon(Icons.person_off_outlined, color: Colors.redAccent),
            ),
            title: const Text('계정 탈퇴', style: TextStyle(color: Colors.redAccent, fontWeight: FontWeight.w600, fontSize: 16)),
            onTap: () => _showDeleteAccountDialog(context),
          ),
        ],
      ),
    );
  }
}
