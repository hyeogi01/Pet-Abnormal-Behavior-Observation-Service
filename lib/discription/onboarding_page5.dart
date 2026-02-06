import 'package:flutter/material.dart';
import 'package:pet_diary/main.dart';

class OnboardingPage5 extends StatelessWidget {
  const OnboardingPage5({super.key});

  @override
  Widget build(BuildContext context) {
    // 오렌지 포인트 컬러 설정
    const Color pointColor = Color(0xFFFF7A00);
    const Color backgroundColor = Color(0xFFF5F5F5);

    return Scaffold(
      backgroundColor: Colors.white,
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 24.0),
          child: Column(
            children: [
              const SizedBox(height: 40),
              // 1. 이미지 영역 (나중에 이미지를 바꿀 수 있는 박스)
              Container(
                width: double.infinity,
                height: MediaQuery.of(context).size.height * 0.45,
                decoration: BoxDecoration(
                  color: backgroundColor,
                  borderRadius: BorderRadius.circular(30),
                ),
                child: Center(
                  // 이 부분을 Image.asset('경로')로 바꾸시면 됩니다.
                  child: Icon(
                      Icons.pets_rounded,
                      size: 100,
                      color: pointColor.withOpacity(0.5)
                  ),
                ),
              ),
              const SizedBox(height: 30),

              // 2. 인디케이터 (현재 2번째 페이지 활성화)
              Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: List.generate(5, (index) => Container(
                  margin: const EdgeInsets.symmetric(horizontal: 4),
                  width: index == 4 ? 40 : 30, // 2번째 바를 길게 표시
                  height: 6,
                  decoration: BoxDecoration(
                    color: index == 4 ? pointColor : Colors.grey.shade300,
                    borderRadius: BorderRadius.circular(3),
                  ),
                )),
              ),
              const SizedBox(height: 40),

              // 3. 텍스트 영역
              const Text(
                "마지막 페이지입니다.",
                style: TextStyle(
                  fontSize: 20,
                  fontWeight: FontWeight.bold,
                  color: Colors.black87,
                ),
              ),
              const SizedBox(height: 12),
              Text(
                "우리 아이의 평소와 다른 움직임을\n실시간으로 분석하여 알려드려요.",
                textAlign: TextAlign.center,
                style: TextStyle(
                  fontSize: 15,
                  color: Colors.grey.shade600,
                  height: 1.5,
                ),
              ),

              const Spacer(),

              // 4. 하단 버튼
              // 4. 하단 버튼
              // 4. 하단 버튼
              SizedBox(
                width: double.infinity,
                height: 56,
                child: ElevatedButton(
                  onPressed: () {
                    // 닫기 버튼 클릭 시 아래에서 로그인 창(Bottom Sheet)이 올라옴
                    showModalBottomSheet(
                      context: context,
                      isScrollControlled: true, // 높이 조절을 위해 true 설정
                      backgroundColor: Colors.transparent, // 모서리 곡률을 위해 투명 설정
                      builder: (context) {
                        return _buildLoginSheet(context);
                      },
                    );
                  },
                  style: ElevatedButton.styleFrom(
                    backgroundColor: pointColor,
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(12),
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

  // --- 로그인 바텀 시트 위젯 ---
  Widget _buildLoginSheet(BuildContext context) {
    return Container(
      height: MediaQuery.of(context).size.height * 0.45,
      padding: const EdgeInsets.all(24),
      decoration: const BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.vertical(top: Radius.circular(25)),
      ),
      child: Column(
        children: [
          // 상단 바 (Handle)
          Container(
            width: 40,
            height: 4,
            margin: const EdgeInsets.only(bottom: 20), // 이 코드로 교체하세요
            decoration: BoxDecoration(
              color: Colors.grey[300],
              borderRadius: BorderRadius.circular(2),
            ),
          ),
          const Text(
            "로그인 및 시작하기",
            style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
          ),
          const SizedBox(height: 30),

          // 카카오 로그인 버튼
          _buildSNSButton(
            icon: Icons.chat_bubble,
            label: "카카오로 시작하기",
            color: const Color(0xFFFEE500),
            textColor: Colors.black,
            onTap: () {
              // 실제 로그인 로직 구현부
            },
          ),
          const SizedBox(height: 12),

          // 구글 로그인 버튼
          _buildSNSButton(
            icon: Icons.g_mobiledata,
            label: "Google로 시작하기",
            color: Colors.white,
            textColor: Colors.black,
            onTap: () {
              // 실제 로그인 로직 구현부
            },
            isBorder: true,
          ),
          const SizedBox(height: 24),

          // 비회원 로그인 서브메뉴
          TextButton(
            onPressed: () {
              // 1. 바텀 시트 닫기
              Navigator.pop(context);
              // 2. 메인 대시보드 화면으로 이동
              Navigator.pushReplacement(
                context,
                MaterialPageRoute(
                  builder: (context) => PetHealthDashboard(),
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

  // SNS 버튼 공통 디자인 위젯
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
        height: 52,
        decoration: BoxDecoration(
          color: color,
          borderRadius: BorderRadius.circular(12),
          border: isBorder ? Border.all(color: Colors.grey.shade300) : null,
        ),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(icon, color: textColor),
            const SizedBox(width: 10),
            Text(
              label,
              style: TextStyle(color: textColor, fontWeight: FontWeight.w600),
            ),
          ],
        ),
      ),
    );
  }
}