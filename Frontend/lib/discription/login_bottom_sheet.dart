import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'package:pet_diary/main.dart';
import 'package:pet_diary/config.dart';
import 'package:flutter_screenutil/flutter_screenutil.dart';
import 'package:pet_diary/mainPage/cam_connect_page.dart';
import 'package:pet_diary/pet_name_input_page.dart';
import 'package:google_sign_in/google_sign_in.dart';
import 'package:firebase_auth/firebase_auth.dart';

// 스와이프 방향에 맞는 슬라이드 전환 라우트
// fromRight: true → 오른쪽에서 슬라이드 인 (다음 페이지)
// fromRight: false → 왼쪽에서 슬라이드 인 (이전 페이지)
Route<void> slideRoute(Widget page, {bool fromRight = true}) {
  return PageRouteBuilder(
    transitionDuration: const Duration(milliseconds: 280),
    pageBuilder: (_, __, ___) => page,
    transitionsBuilder: (_, animation, __, child) {
      final begin = fromRight ? const Offset(1.0, 0.0) : const Offset(-1.0, 0.0);
      return SlideTransition(
        position: Tween(begin: begin, end: Offset.zero)
            .chain(CurveTween(curve: Curves.easeInOut))
            .animate(animation),
        child: child,
      );
    },
  );
}

void showLoginBottomSheet(BuildContext context) {
  showModalBottomSheet(
    context: context,
    isScrollControlled: true,
    backgroundColor: Colors.transparent,
    builder: (_) => const _LoginSheet(),
  );
}

class _LoginSheet extends StatefulWidget {
  const _LoginSheet();

  @override
  State<_LoginSheet> createState() => _LoginSheetState();
}

class _LoginSheetState extends State<_LoginSheet> {
  Future<void> _handleGoogleSignIn() async {
    // async 전에 미리 캡처 — popUntil 이후 context가 unmount돼도 안전
    final navigator = Navigator.of(context);
    final messenger = ScaffoldMessenger.of(context);

    try {
      final googleSignIn = GoogleSignIn();
      await googleSignIn.signOut();
      final googleUser = await googleSignIn.signIn();
      if (googleUser == null) return;

      final googleAuth = await googleUser.authentication;
      final credential = GoogleAuthProvider.credential(
        accessToken: googleAuth.accessToken,
        idToken: googleAuth.idToken,
      );

      final userCredential =
          await FirebaseAuth.instance.signInWithCredential(credential);
      final idToken = await userCredential.user!.getIdToken();

      final response = await http.post(
        Uri.parse('${Config.apiBaseUrl}/auth/google/'),
        headers: {...Config.ngrokHeaders, 'Content-Type': 'application/json'},
        body: jsonEncode({'id_token': idToken}),
      ).timeout(const Duration(seconds: 10));

      final result = jsonDecode(response.body);

      if (response.statusCode == 200 && result['status'] == 'success') {
        navigator.popUntil((route) => route.isFirst);
        if (result['has_pet_info']) {
          navigator.pushReplacement(
              MaterialPageRoute(builder: (_) => PetHealthDashboard(userId: result['user_id'])));
        } else {
          navigator.pushReplacement(
              MaterialPageRoute(builder: (_) => PetNameInputPage(userId: result['user_id'])));
        }
      } else {
        messenger.showSnackBar(
          SnackBar(content: Text("구글 로그인 실패: ${result['message'] ?? '알 수 없는 오류'}")),
        );
      }
    } catch (e) {
      messenger.showSnackBar(
        SnackBar(content: Text("구글 로그인 오류: $e")),
      );
    }
  }

  Future<void> _handleAuth(String id, String pw, bool isSignup,
      void Function(bool) setLoading) async {
    if (id.isEmpty || pw.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text("아이디와 비밀번호를 입력해주세요.")),
      );
      return;
    }
    // async 전에 미리 캡처
    final navigator = Navigator.of(context);
    final messenger = ScaffoldMessenger.of(context);
    final endpoint = isSignup ? '/signup/' : '/login/';

    try {
      final response = await http.post(
        Uri.parse('${Config.apiBaseUrl}$endpoint'),
        headers: Config.ngrokHeaders,
        body: jsonEncode({'user_id': id, 'password': pw}),
      ).timeout(const Duration(seconds: 10));

      final result = jsonDecode(response.body);

      if (response.statusCode == 200 && result['status'] == 'success') {
        navigator.popUntil((route) => route.isFirst);
        if (isSignup) {
          navigator.pushReplacement(
              MaterialPageRoute(builder: (_) => PetNameInputPage(userId: id)));
        } else {
          if (result['has_pet_info'] ?? false) {
            navigator.pushReplacement(
                MaterialPageRoute(builder: (_) => PetHealthDashboard(userId: result['user_id'])));
          } else {
            navigator.pushReplacement(
                MaterialPageRoute(builder: (_) => PetNameInputPage(userId: id)));
          }
        }
      } else {
        messenger.showSnackBar(
          SnackBar(content: Text("${isSignup ? '회원가입' : '로그인'} 실패: ${result['message'] ?? '알 수 없는 오류'}")),
        );
      }
    } catch (e) {
      messenger.showSnackBar(
        const SnackBar(content: Text("서버 연결 실패: 백엔드가 켜져있는지 확인해주세요.")),
      );
    }
  }

  void _showEmailAuthDialog() {
    final idController = TextEditingController();
    final pwController = TextEditingController();

    showDialog(
      context: context,
      builder: (ctx) => StatefulBuilder(
        builder: (ctx, setDialogState) {
          bool isLoading = false;
          return AlertDialog(
            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
            title: const Text("이메일 로그인/가입", style: TextStyle(fontWeight: FontWeight.bold)),
            content: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                if (isLoading)
                  const Padding(
                    padding: EdgeInsets.symmetric(vertical: 20),
                    child: CircularProgressIndicator(),
                  )
                else ...[
                  TextField(controller: idController, decoration: const InputDecoration(hintText: "아이디")),
                  const SizedBox(height: 10),
                  TextField(controller: pwController, obscureText: true, decoration: const InputDecoration(hintText: "비밀번호")),
                ],
              ],
            ),
            actions: isLoading
                ? []
                : [
                    TextButton(
                      onPressed: () async {
                        setDialogState(() => isLoading = true);
                        await _handleAuth(idController.text, pwController.text, false,
                            (v) => setDialogState(() => isLoading = v));
                        if (mounted) setDialogState(() => isLoading = false);
                      },
                      child: Text("로그인", style: TextStyle(color: Theme.of(context).primaryColor)),
                    ),
                    ElevatedButton(
                      onPressed: () async {
                        setDialogState(() => isLoading = true);
                        await _handleAuth(idController.text, pwController.text, true,
                            (v) => setDialogState(() => isLoading = v));
                        if (mounted) setDialogState(() => isLoading = false);
                      },
                      style: ElevatedButton.styleFrom(elevation: 0),
                      child: const Text("회원가입", style: TextStyle(color: Colors.white)),
                    ),
                  ],
          );
        },
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
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
            width: 40.w, height: 4.h,
            margin: EdgeInsets.only(bottom: 24.h),
            decoration: BoxDecoration(
              color: Colors.grey[300],
              borderRadius: BorderRadius.circular(2.r),
            ),
          ),
          const Text("로그인 및 시작하기",
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
          SizedBox(height: 20.h),
          _buildSNSButton(
            icon: Icons.chat_bubble, label: "카카오로 시작하기",
            color: const Color(0xFFFEE500), textColor: Colors.black,
            onTap: () {},
          ),
          SizedBox(height: 10.h),
          _buildSNSButton(
            icon: Icons.g_mobiledata, label: "Google로 시작하기",
            color: Colors.white, textColor: Colors.black,
            onTap: _handleGoogleSignIn, isBorder: true,
          ),
          SizedBox(height: 10.h),
          _buildSNSButton(
            icon: Icons.email, label: "이메일로 시작하기",
            color: Colors.grey[200]!, textColor: Colors.black,
            onTap: () => _showEmailAuthDialog(),
          ),
          TextButton(
            onPressed: () {
              final navigator = Navigator.of(context);
              navigator.pop();
              navigator.pushReplacement(
                  MaterialPageRoute(builder: (_) => const PetNameInputPage(userId: "guest_user")));
            },
            child: Text("로그인 없이 둘러보기 (비회원)",
                style: TextStyle(color: Colors.grey[500],
                    decoration: TextDecoration.underline, fontSize: 14)),
          ),
          SizedBox(height: 8.h),
          TextButton.icon(
            onPressed: () {
              final navigator = Navigator.of(context);
              navigator.pop();
              navigator.push(
                  MaterialPageRoute(builder: (_) => const CamConnectPage()));
            },
            icon: const Icon(Icons.videocam, color: Colors.orange, size: 20),
            label: const Text("공기계를 CCTV로 사용하기",
                style: TextStyle(color: Colors.orange,
                    fontWeight: FontWeight.bold, fontSize: 15)),
          ),
        ],
      ),
    );
  }

  Widget _buildSNSButton({
    required IconData icon, required String label,
    required Color color, required Color textColor,
    required VoidCallback onTap, bool isBorder = false,
  }) {
    return InkWell(
      onTap: onTap,
      child: Container(
        width: double.infinity, height: 48.h,
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
            Text(label, style: TextStyle(color: textColor,
                fontWeight: FontWeight.w600, fontSize: 15)),
          ],
        ),
      ),
    );
  }
}
