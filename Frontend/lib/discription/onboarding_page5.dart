import 'package:flutter/material.dart';
import 'package:pet_diary/pet_name_input_page.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'package:pet_diary/main.dart';
import 'package:flutter_screenutil/flutter_screenutil.dart';

class OnboardingPage5 extends StatefulWidget {
  const OnboardingPage5({super.key});

  @override
  State<OnboardingPage5> createState() => _OnboardingPage5State();
}

class _OnboardingPage5State extends State<OnboardingPage5> {
  bool _isLoading = false;

  // 1. 서버와 통신하는 실제 로직 (로그인/회원가입만 신속히 처리)
  Future<void> _handleAuth(BuildContext dialogContext, String id, String pw, bool isSignup) async {
    if (id.isEmpty || pw.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text("아이디와 비밀번호를 입력해주세요.")),
      );
      return;
    }

    final String endpoint = isSignup ? '/signup/' : '/login/';
    final Uri url = Uri.parse('http://localhost:8080$endpoint');

    // 다이얼로그 내부 로딩 상태 활성화 (StatefulBuilder 내의 setState 사용을 위해 아래 다이얼로그 로직에서 처리)
    
    try {
      final response = await http.post(
        url,
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({'user_id': id, 'password': pw}),
      ).timeout(const Duration(seconds: 10));

      final result = jsonDecode(response.body);
      
      if (response.statusCode == 200 && result['status'] == 'success') {
        if (!mounted) return;
        
        // 커밋 전 모든 다이얼로그와 바텀시트 닫기
        Navigator.of(context).popUntil((route) => route.isFirst || route.settings.name == '/');

        if (isSignup) {
          Navigator.pushReplacement(
            context,
            MaterialPageRoute(builder: (context) => PetNameInputPage(userId: id)),
          );
        } else {
          bool hasPet = result['has_pet_info'] ?? false;
          if (hasPet) {
            Navigator.pushReplacement(
              context,
              MaterialPageRoute(
                builder: (context) => PetHealthDashboard(userId: result['user_id']),
              ),
            );
          } else {
            Navigator.pushReplacement(
              context,
              MaterialPageRoute(
                builder: (context) => PetNameInputPage(userId: id),
              ),
            );
          }
        }
      } else {
        if (!mounted) return;
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text("${isSignup ? '회원가입' : '로그인'} 실패: ${result['message'] ?? '알 수 없는 오류'}")),
        );
      }
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text("서버 연결 실패: 백엔드가 켜져있는지 확인해주세요.")),
      );
    }
  }

  void _showEmailAuthDialog(BuildContext context) {
    final TextEditingController idController = TextEditingController();
    final TextEditingController pwController = TextEditingController();
    bool isDialogLoading = false;

    showDialog(
      context: context,
      builder: (context) => StatefulBuilder(
        builder: (context, setDialogState) {
          return AlertDialog(
            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
            title: const Text("이메일 로그인/가입", style: TextStyle(fontWeight: FontWeight.bold)),
            content: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                if (isDialogLoading)
                  const Padding(
                    padding: EdgeInsets.symmetric(vertical: 20),
                    child: CircularProgressIndicator(color: Color(0xFFFF7A00)),
                  )
                else ...[
                  TextField(controller: idController, decoration: const InputDecoration(hintText: "아이디")),
                  const SizedBox(height: 10),
                  TextField(controller: pwController, obscureText: true, decoration: const InputDecoration(hintText: "비밀번호")),
                ],
              ],
            ),
            actions: isDialogLoading 
              ? [] 
              : [
                  TextButton(
                    onPressed: () async {
                      setDialogState(() => isDialogLoading = true);
                      await _handleAuth(context, idController.text, pwController.text, false);
                      if (mounted) setDialogState(() => isDialogLoading = false);
                    },
                    child: const Text("로그인", style: TextStyle(color: Colors.orange)),
                  ),
                  ElevatedButton(
                    onPressed: () async {
                      setDialogState(() => isDialogLoading = true);
                      await _handleAuth(context, idController.text, pwController.text, true);
                      if (mounted) setDialogState(() => isDialogLoading = false);
                    },
                    style: ElevatedButton.styleFrom(backgroundColor: Colors.orange, elevation: 0),
                    child: const Text("회원가입", style: TextStyle(color: Colors.white)),
                  ),
                ],
          );
        }
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    const Color pointColor = Color(0xFFFF7A00);
    const Color backgroundColor = Color(0xFFF5F5F5);

    return Scaffold(
      backgroundColor: Colors.white,
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 24),
          child: Column(
            children: [
              const SizedBox(height: 40),
              Container(
                width: double.infinity,
                height: MediaQuery.of(context).size.height * 0.45,
                decoration: BoxDecoration(
                  color: backgroundColor,
                  borderRadius: BorderRadius.circular(30.r),
                ),
                child: Center(
                  child: Icon(
                      Icons.pets_rounded,
                      size: 100,
                      color: pointColor.withOpacity(0.5)
                  ),
                ),
              ),
              const SizedBox(height: 30),
              Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: List.generate(5, (index) => Container(
                  margin: const EdgeInsets.symmetric(horizontal: 4),
                  width: index == 4 ? 40 : 30,
                  height: 6,
                  decoration: BoxDecoration(
                    color: index == 4 ? pointColor : Colors.grey.shade300,
                    borderRadius: BorderRadius.circular(3.r),
                  ),
                )),
              ),
              const SizedBox(height: 40),
              const Text(
                "마지막 페이지입니다.",
                style: TextStyle(
                  fontSize: 20,
                  fontWeight: FontWeight.bold,
                  color: Colors.black87,
                ),
              ),
              const SizedBox(height: 12),
              const Text(
                "우리 아이의 평소와 다른 움직임을\n실시간으로 분석하여 알려드려요.",
                textAlign: TextAlign.center,
                style: TextStyle(
                  fontSize: 15,
                  color: Colors.grey,
                  height: 1.5,
                ),
              ),
              const Spacer(),
              SizedBox(
                width: double.infinity,
                height: 56,
                child: ElevatedButton(
                  onPressed: () {
                    showModalBottomSheet(
                      context: context,
                      isScrollControlled: true,
                      backgroundColor: Colors.transparent,
                      builder: (context) => _buildLoginSheet(context),
                    );
                  },
                  style: ElevatedButton.styleFrom(
                    backgroundColor: pointColor,
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(12.r),
                    ),
                    elevation: 0,
                  ),
                  child: const Text(
                    "닫기",
                    style: TextStyle(
                      fontSize: 18,
                      fontWeight: FontWeight.bold,
                      color: Colors.white,
                    ),
                  ),
                ),
              ),
              const SizedBox(height: 20),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildLoginSheet(BuildContext context) {
    return Container(
      height: MediaQuery.of(context).size.height * 0.50,
      padding: EdgeInsets.symmetric(horizontal: 24.w, vertical: 20.h),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.vertical(top: Radius.circular(20.r)),
      ),
      child: Column(
        children: [
          Container(
            width: 40.w,
            height: 4.h,
            margin: EdgeInsets.only(bottom: 24.h),
            decoration: BoxDecoration(
              color: Colors.grey[300],
              borderRadius: BorderRadius.circular(2.r),
            ),
          ),
          const Text(
            "로그인 및 시작하기",
            style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
          ),
          SizedBox(height: 20.h),
          _buildSNSButton(
            icon: Icons.chat_bubble,
            label: "카카오로 시작하기",
            color: const Color(0xFFFEE500),
            textColor: Colors.black,
            onTap: () {},
          ),
          SizedBox(height: 10.h),
          _buildSNSButton(
            icon: Icons.g_mobiledata,
            label: "Google로 시작하기",
            color: Colors.white,
            textColor: Colors.black,
            onTap: () {},
            isBorder: true,
          ),
          SizedBox(height: 10.h),
          _buildSNSButton(
            icon: Icons.email,
            label: "이메일로 시작하기",
            color: Colors.grey[200]!,
            textColor: Colors.black,
            onTap: () {
              Navigator.pop(context);
              _showEmailAuthDialog(context);
            },
          ),
          TextButton(
            onPressed: () {
              Navigator.pop(context);
              Navigator.pushReplacement(
                context,
                MaterialPageRoute(
                  builder: (context) => const PetNameInputPage(userId: "guest_user"),
                ),
              );
            },
            child: Text(
              "로그인 없이 둘러보기 (비회원)",
              style: TextStyle(
                color: Colors.grey[500],
                decoration: TextDecoration.underline,
                fontSize: 14,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildSNSButton({
    required IconData icon,
    required String label,
    required Color color,
    required Color textColor,
    required VoidCallback onTap,
    bool isBorder = false,
  }) {
    return InkWell(
      onTap: onTap,
      child: Container(
        width: double.infinity,
        height: 48.h,
        decoration: BoxDecoration(
          color: color,
          borderRadius: BorderRadius.circular(12.r),
          border: isBorder ? Border.all(color: Colors.grey.shade300) : null,
        ),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(icon, color: textColor, size: 22),
            SizedBox(width: 10.w),
            Text(
              label,
              style: TextStyle(color: textColor, fontWeight: FontWeight.w600, fontSize: 15),
            ),
          ],
        ),
      ),
    );
  }
}