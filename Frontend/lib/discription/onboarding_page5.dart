import 'package:flutter/material.dart';
import 'package:pet_diary/discription/onboarding_page4.dart';
import 'package:pet_diary/discription/login_bottom_sheet.dart';

class OnboardingPage5 extends StatefulWidget {
  const OnboardingPage5({super.key});

  @override
  State<OnboardingPage5> createState() => _OnboardingPage5State();
}

class _OnboardingPage5State extends State<OnboardingPage5>
    with TickerProviderStateMixin {
  late final AnimationController _floatControllerY;
  late final AnimationController _floatControllerX;
  late final Animation<double> _floatAnimY;
  late final Animation<double> _floatAnimX;

  @override
  void initState() {
    super.initState();
    _floatControllerY = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1700),
    )..repeat(reverse: true);
    _floatAnimY = Tween<double>(begin: 0.0, end: 1.0).animate(
      CurvedAnimation(parent: _floatControllerY, curve: Curves.easeInOut),
    );
    _floatControllerX = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 2300),
    )..repeat(reverse: true);
    _floatAnimX = Tween<double>(begin: 0.0, end: 1.0).animate(
      CurvedAnimation(parent: _floatControllerX, curve: Curves.easeInOut),
    );
  }

  @override
  void dispose() {
    _floatControllerY.dispose();
    _floatControllerX.dispose();
    super.dispose();
  }

  void _onSwipe(DragEndDetails details) {
    if (details.primaryVelocity == null) return;
    if (details.primaryVelocity! < -300) {
      // 마지막 페이지에서 왼쪽 스와이프 → 로그인 바텀시트
      showLoginBottomSheet(context);
    } else if (details.primaryVelocity! > 300) {
      Navigator.pushReplacement(
          context, slideRoute(const OnboardingPage4(), fromRight: false));
    }
  }

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onHorizontalDragEnd: _onSwipe,
      child: Scaffold(
        backgroundColor: Colors.white,
        body: SafeArea(
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 24.0),
            child: Column(
              children: [
                const SizedBox(height: 40),
                SizedBox(
                  width: double.infinity,
                  height: MediaQuery.of(context).size.height * 0.45,
                  child: Stack(
                    children: [
                      Container(
                        width: double.infinity,
                        height: double.infinity,
                        decoration: BoxDecoration(
                          color: const Color(0xFF34D399).withOpacity(0.08),
                          borderRadius: BorderRadius.circular(28),
                        ),
                        clipBehavior: Clip.hardEdge,
                        child: Image.asset(
                          'assets/images/intro_settings.png',
                          fit: BoxFit.contain,
                          errorBuilder: (_, __, ___) => const Center(
                            child: Icon(Icons.directions_walk_rounded, size: 100,
                                color: Color(0x6634D399)),
                          ),
                        ),
                      ),
                      Positioned.fill(
                        child: LayoutBuilder(
                          builder: (context, constraints) {
                            const charSize = 80.0;
                            const padding = 12.0;
                            final maxX = constraints.maxWidth - charSize - padding * 2;
                            final maxY = constraints.maxHeight - charSize - padding * 2;
                            return AnimatedBuilder(
                              animation: Listenable.merge([_floatAnimX, _floatAnimY]),
                              builder: (_, __) => Stack(
                                children: [
                                  Positioned(
                                    left: padding + _floatAnimX.value * maxX,
                                    top: padding + _floatAnimY.value * maxY,
                                    child: Image.asset(
                                      'assets/images/pet_character.png',
                                      width: charSize, height: charSize, fit: BoxFit.contain,
                                      errorBuilder: (_, __, ___) => const SizedBox.shrink(),
                                    ),
                                  ),
                                ],
                              ),
                            );
                          },
                        ),
                      ),
                    ],
                  ),
                ),
                const SizedBox(height: 24),
                Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: List.generate(5, (i) => AnimatedContainer(
                    duration: const Duration(milliseconds: 250),
                    margin: const EdgeInsets.symmetric(horizontal: 4),
                    width: i == 4 ? 28 : 8, height: 8,
                    decoration: BoxDecoration(
                      color: i == 4 ? const Color(0xFF34D399) : Colors.grey.shade300,
                      borderRadius: BorderRadius.circular(4),
                    ),
                  )),
                ),
                const SizedBox(height: 32),
                const Text('산책 · 체중 관리',
                    style: TextStyle(fontSize: 22, fontWeight: FontWeight.bold,
                        color: Colors.black87),
                    textAlign: TextAlign.center),
                const SizedBox(height: 12),
                Text('산책 기록과 체중 변화를 한눈에 확인하고\n건강을 꾸준히 챙겨요',
                    textAlign: TextAlign.center,
                    style: TextStyle(fontSize: 15, color: Colors.grey.shade600,
                        height: 1.6)),
                const Spacer(),
                Row(
                  children: [
                    TextButton(
                      onPressed: () => showLoginBottomSheet(context),
                      child: Text('건너뛰기',
                          style: TextStyle(color: Colors.grey.shade500)),
                    ),
                    const Spacer(),
                    SizedBox(
                      width: 130, height: 48,
                      child: ElevatedButton(
                        onPressed: () => showLoginBottomSheet(context),
                        style: ElevatedButton.styleFrom(
                          backgroundColor: const Color(0xFF34D399),
                          foregroundColor: Colors.white,
                          shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(14)),
                          elevation: 0,
                        ),
                        child: const Text('시작하기',
                            style: TextStyle(fontSize: 15,
                                fontWeight: FontWeight.bold)),
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 20),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
