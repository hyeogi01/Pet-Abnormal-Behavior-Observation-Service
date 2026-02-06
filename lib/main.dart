import 'package:flutter/material.dart';
// 온보딩 페이지 경로는 프로젝트 환경에 맞게 유지하세요.
// import 'package:pet_diary/discription/onboarding_page.dart';
import 'package:pet_diary/mainPage/odd_pet.dart';
void main() {
  runApp(const MaterialApp(
    home: PetHealthDashboard(),
    debugShowCheckedModeBanner: false,
  ));
}

class PetHealthDashboard extends StatefulWidget {
  const PetHealthDashboard({super.key});

  @override
  State<PetHealthDashboard> createState() => _PetHealthDashboardState();
}

class _PetHealthDashboardState extends State<PetHealthDashboard> {
  // 현재 선택된 메뉴 인덱스 (기본값 홈 = 2)
  int _selectedIndex = 2;

  // 하단 탭 클릭 시 상태 변경 함수
  void _onItemTapped(int index) {
    setState(() {
      _selectedIndex = index;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.white,
      appBar: AppBar(
        backgroundColor: Colors.white,
        elevation: 0,
        leading: const Icon(Icons.menu, color: Colors.black),
        title: const Text('Daily Behavior Diary',
            style: TextStyle(color: Colors.black, fontSize: 14, fontWeight: FontWeight.bold)),
        centerTitle: true,
        actions: [IconButton(icon: const Icon(Icons.share, color: Colors.blue), onPressed: () {})],
      ),

      // 선택된 탭 인덱스에 따라 홈 화면 또는 준비중 화면 표시
      body: _selectedIndex == 2
          ? _buildDashboardHome()
          : Center(child: Text('준비 중인 페이지입니다.', style: TextStyle(color: Colors.grey[400], fontSize: 16))),

      // 하단 내비게이션 바
      bottomNavigationBar: BottomNavigationBar(
        type: BottomNavigationBarType.fixed,
        backgroundColor: Colors.white,
        currentIndex: _selectedIndex,
        onTap: _onItemTapped,
        selectedItemColor: Colors.amber[700], // 현재 페이지 노란색(강조)
        unselectedItemColor: Colors.grey[400],
        selectedFontSize: 11,
        unselectedFontSize: 11,
        items: const [
          BottomNavigationBarItem(icon: Icon(Icons.stars), label: '모니터링'),
          BottomNavigationBarItem(icon: Icon(Icons.circle_outlined), label: '미정'),
          BottomNavigationBarItem(icon: Icon(Icons.home), label: '홈'),
          BottomNavigationBarItem(icon: Icon(Icons.favorite), label: '사진첩'),
          BottomNavigationBarItem(icon: Icon(Icons.person), label: '마이페이지'),
        ],
      ),
    );
  }

  // --- 메인 홈 대시보드 UI ---
  Widget _buildDashboardHome() {
    return SingleChildScrollView(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _buildHeaderCard(),
          const SizedBox(height: 16),
          Row(
            children: [
              _buildActionButton(Icons.book, '일상 일기', '기분 & 활동량', Colors.blue,
                      () => Navigator.push(context, MaterialPageRoute(builder: (context) => const PageA()))),
              const SizedBox(width: 12),
              _buildActionButton(Icons.error_outline, '이상 행동', '건강 체크', Colors.orange,
                      () => Navigator.push(context, MaterialPageRoute(builder: (context) => PageB()))),
            ],
          ),
          const SizedBox(height: 24),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              const Text('최근 일기', style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
              Text('전체보기 →', style: TextStyle(color: Colors.purple[300], fontSize: 12)),
            ],
          ),
          const SizedBox(height: 12),
          _buildDiaryItem('2026년 2월 6일', '목요일', 85, true),
          _buildDiaryItem('2026년 2월 5일', '수요일', 72, false),
          _buildDiaryItem('2026년 2월 4일', '화요일', 90, false),
          const SizedBox(height: 24),
          _buildTrendSection(),
          const SizedBox(height: 32),
          const Center(
            child: Column(
              children: [
                Text('AI가 24시간 콩이를 모니터링하고 있어요', style: TextStyle(color: Colors.grey, fontSize: 12)),
                SizedBox(height: 4),
                Text('8가지 데이터셋 기반 건강 분석 시스템', style: TextStyle(color: Colors.grey, fontSize: 11)),
              ],
            ),
          ),
          const SizedBox(height: 40),
        ],
      ),
    );
  }

  // --- 헬퍼 함수들 ---

  Widget _buildHeaderCard() {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(20),
        gradient: const LinearGradient(colors: [Colors.purple, Colors.orangeAccent]),
      ),
      child: Column(
        children: [
          const Row(
            children: [
              CircleAvatar(radius: 25, backgroundColor: Colors.white),
              SizedBox(width: 12),
              Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text('콩이의 건강일기', style: TextStyle(color: Colors.white, fontSize: 20, fontWeight: FontWeight.bold)),
                  Text('AI 기반 반려동물 케어', style: TextStyle(color: Colors.white70, fontSize: 12)),
                ],
              )
            ],
          ),
          const SizedBox(height: 20),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceAround,
            children: [
              _buildStatItem('12', '총 일기'),
              _buildStatItem('85', '평균 활동'),
              _buildStatItem('98%', '건강도'),
            ],
          )
        ],
      ),
    );
  }

  Widget _buildStatItem(String value, String label) {
    return Column(
      children: [
        Text(value, style: const TextStyle(color: Colors.white, fontSize: 18, fontWeight: FontWeight.bold)),
        Text(label, style: const TextStyle(color: Colors.white70, fontSize: 12)),
      ],
    );
  }

  Widget _buildActionButton(IconData icon, String title, String subTitle, Color color, VoidCallback onTap) {
    return Expanded(
      child: InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(15),
        child: Container(
          padding: const EdgeInsets.symmetric(vertical: 20),
          decoration: BoxDecoration(color: color, borderRadius: BorderRadius.circular(15)),
          child: Column(
            children: [
              Icon(icon, color: Colors.white, size: 30),
              const SizedBox(height: 8),
              Text(title, style: const TextStyle(color: Colors.white, fontWeight: FontWeight.bold)),
              Text(subTitle, style: const TextStyle(color: Colors.white70, fontSize: 10)),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildDiaryItem(String date, String day, int activity, bool hasWarning) {
    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(15),
        boxShadow: const [BoxShadow(color: Colors.black12, blurRadius: 4)],
      ),
      child: Row(
        children: [
          Container(width: 50, height: 50, decoration: BoxDecoration(color: Colors.grey[300], borderRadius: BorderRadius.circular(8))),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(date, style: const TextStyle(fontWeight: FontWeight.bold)),
                Text(day, style: const TextStyle(color: Colors.grey, fontSize: 12)),
                Row(
                  children: [
                    const Icon(Icons.trending_up, size: 14, color: Colors.green),
                    Text(' 활동 $activity', style: const TextStyle(fontSize: 12)),
                    if (hasWarning) ...[
                      const SizedBox(width: 8),
                      Container(
                        padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                        decoration: BoxDecoration(color: Colors.orange[100], borderRadius: BorderRadius.circular(4)),
                        child: const Text('주의사항', style: TextStyle(color: Colors.orange, fontSize: 10)),
                      )
                    ]
                  ],
                )
              ],
            ),
          ),
          const Icon(Icons.sentiment_satisfied_alt, color: Colors.orange),
        ],
      ),
    );
  }

  Widget _buildTrendSection() {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(15),
        boxShadow: const [BoxShadow(color: Colors.black12, blurRadius: 4)],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Row(
            children: [
              Icon(Icons.trending_up, color: Colors.green, size: 20),
              SizedBox(width: 8),
              Text('이번 주 건강 트렌드', style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16)),
            ],
          ),
          const SizedBox(height: 16),
          _buildTrendRow('평균 활동량', 0.82, Colors.green, '82%'),
          _buildTrendRow('체중 관리', 0.95, Colors.blue, '95%'),
          _buildTrendRow('스트레스 관리', 0.88, Colors.purple, '88%'),
          const SizedBox(height: 16),
          Container(
            width: double.infinity,
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: Colors.green[50],
              borderRadius: BorderRadius.circular(10),
              border: Border.all(color: Colors.green[100]!),
            ),
            child: const Text(
              '🎉 콩이는 이번 주 매우 건강하게 지냈어요! 활동량과 식사 패턴이 안정적입니다.',
              style: TextStyle(color: Colors.green, fontSize: 13),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildTrendRow(String label, double value, Color color, String percent) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 8.0),
      child: Row(
        children: [
          Expanded(flex: 3, child: Text(label, style: const TextStyle(fontSize: 13))),
          Expanded(
            flex: 7,
            child: ClipRRect(
              borderRadius: BorderRadius.circular(10),
              child: LinearProgressIndicator(
                value: value,
                backgroundColor: Colors.grey[200],
                valueColor: AlwaysStoppedAnimation<Color>(color),
                minHeight: 8,
              ),
            ),
          ),
          const SizedBox(width: 10),
          Text(percent, style: TextStyle(fontSize: 12, color: color, fontWeight: FontWeight.bold)),
        ],
      ),
    );
  }
}

// 임시 페이지 클래스
class PageA extends StatelessWidget { const PageA({super.key}); @override Widget build(BuildContext context) => Scaffold(appBar: AppBar(title: const Text('일상 일기'))); }
//class PageB extends StatelessWidget { const PageB({super.key}); @override Widget build(BuildContext context) => Scaffold(appBar: AppBar(title: const Text('이상 행동'))); }