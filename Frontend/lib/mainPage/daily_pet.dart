import 'package:flutter/material.dart';
import 'package:pet_diary/mainPage/pet_activity.dart';

class daily_pet extends StatelessWidget {
  const daily_pet({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFFF9F9F9),
      // 상단 앱바 (이미지의 오렌지색 헤더 부분)
      appBar: AppBar(
        backgroundColor: Colors.orange,
        elevation: 0,
        leading: IconButton(
          icon: const Icon(Icons.arrow_back, color: Colors.white),
          onPressed: () => Navigator.pop(context),
        ),
        title: const Text('일상 행동 일기', style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold)),
        centerTitle: true,
      ),
      body: SingleChildScrollView(
        child: Column(
          children: [
            // 1. 헤더 하단 정보 (날짜 등)
            Container(
              width: double.infinity,
              color: Colors.orange,
              padding: const EdgeInsets.fromLTRB(16, 0, 16, 20),
              child: const Text('2026년 2월 6일 목요일', style: TextStyle(color: Colors.white70, fontSize: 12)),
            ),

            Padding(
              padding: const EdgeInsets.all(16.0),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  // 2. 기분 섹션
                  _buildSectionTitle('🐾 오늘 하루 콩이의 기분!'),
                  _buildMoodCard(),
                  const SizedBox(height: 24),

                  // 3. 오늘의 순간들 (갤러리)
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      _buildSectionTitle('📸 오늘의 순간들'),
                      const Text('대표 사진', style: TextStyle(color: Colors.orange, fontSize: 10)),
                    ],
                  ),
                  _buildPhotoGallery(),
                  const SizedBox(height: 24),

                  // 4. 무슨 일이 있었어? (AI 요약)
                  _buildSectionTitle('💬 쫑이야 오늘은 어땠어?'),
                  _buildAISummaryCard(),
                  const SizedBox(height: 24),

                  // 5. 펫페오톡 선생님 조언 (보라색 카드)
                  _buildTeacherAdviceCard(),
                  const SizedBox(height: 24),

                  // 6. 보호자 메모
                  _buildSectionTitle('보호자 메모'),
                  _buildMemoField(),
                  const SizedBox(height: 30),

                  // 7. 하단 버튼
                  _buildBottomButton(context),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  // --- 소제목 위젯 ---
  Widget _buildSectionTitle(String title) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: Text(title, style: const TextStyle(fontSize: 15, fontWeight: FontWeight.bold, color: Colors.black87)),
    );
  }

  // --- 기분 카드 ---
  Widget _buildMoodCard() {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFFFFF8F0),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          const Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text('평균 기분', style: TextStyle(color: Colors.grey, fontSize: 12)),
              Text('즐거움', style: TextStyle(color: Colors.orange, fontSize: 20, fontWeight: FontWeight.bold)),
            ],
          ),
          Icon(Icons.sentiment_satisfied_alt, color: Colors.orange[300], size: 40),
        ],
      ),
    );
  }

  // --- 사진 갤러리 (가로 스크롤) ---
  Widget _buildPhotoGallery() {
    return Column(
      children: [
        // 1. 대표 사진 (상단)
        Container(
          height: 220,
          width: double.infinity,
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(15),
            color: Colors.grey[300],
            image: const DecorationImage(
              image: NetworkImage('https://via.placeholder.com/600x400'),
              fit: BoxFit.cover,
            ),
          ),
        ),
        const SizedBox(height: 12),

        // 2. 하단 정사각형 4개 (테두리 제거됨)
        GridView.builder(
          shrinkWrap: true,
          physics: const NeverScrollableScrollPhysics(),
          itemCount: 4,
          gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
            crossAxisCount: 4,     // 가로 4칸 고정
            crossAxisSpacing: 8,   // 간격
            mainAxisSpacing: 8,
            childAspectRatio: 1,   // 1:1 정사각형 비율
          ),
          itemBuilder: (context, index) {
            return Container(
              decoration: BoxDecoration(
                borderRadius: BorderRadius.circular(12),
                color: Colors.grey[300],
                // border 속성을 삭제하여 노란색/주황색 체크 라인을 없앴습니다.
                image: const DecorationImage(
                  image: NetworkImage('https://via.placeholder.com/150'),
                  fit: BoxFit.cover,
                ),
              ),
            );
          },
        ),
      ],
    );
  }

  // --- AI 요약 카드 ---
  Widget _buildAISummaryCard() {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFFF0F5FF),
        borderRadius: BorderRadius.circular(12),
      ),
      child: const Text(
        '오늘 콩이는 아침 8시 30분에 맛있게 밥을 먹었어요! 그리고 10시에는 공원에서 신나게 뛰어놀았네요. 오후에는 편안하게 낮잠을 자고...',
        style: TextStyle(color: Colors.blueGrey, fontSize: 13, height: 1.5),
      ),
    );
  }

  // --- 펫페오톡 선생님 조언 (보라색) ---
  Widget _buildTeacherAdviceCard() {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        gradient: const LinearGradient(colors: [Colors.purpleAccent, Colors.pinkAccent]),
        borderRadius: BorderRadius.circular(15),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Row(
            children: [
              CircleAvatar(radius: 15, backgroundColor: Colors.white, child: Icon(Icons.person, size: 20)),
              SizedBox(width: 10),
              Text('쫑이의 하루!', style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold)),
            ],
          ),
          const SizedBox(height: 12),
          Container(
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: Colors.white.withOpacity(0.2),
              borderRadius: BorderRadius.circular(10),
            ),
            child: const Text(
              '콩이가 오늘 활발하게 활동했네요! 규칙적인 식사와 충분한 운동, 휴식이 잘 이루어지고 있습니다.',
              style: TextStyle(color: Colors.white, fontSize: 12),
            ),
          ),
        ],
      ),
    );
  }

  // --- 메모 입력 필드 ---
  Widget _buildMemoField() {
    return TextField(
      maxLines: 4,
      decoration: InputDecoration(
        hintText: '오늘 반려동물과 함께한 특별한 순간을 기록해주세요...',
        hintStyle: const TextStyle(fontSize: 13, color: Colors.grey),
        filled: true,
        fillColor: Colors.white,
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(12),
          borderSide: const BorderSide(color: Colors.black12),
        ),
        enabledBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(12),
          borderSide: const BorderSide(color: Colors.black12),
        ),
      ),
    );
  }

  // --- 하단 주황색 버튼 ---
  Widget _buildBottomButton(BuildContext context) {
    return SizedBox(
      width: double.infinity,
      height: 50,
      child: ElevatedButton(
        onPressed: () {
          // pet_activity 페이지로 이동
          Navigator.push(
            context,
            MaterialPageRoute(builder: (context) => PetActivityPage()),
          );
        },
        style: ElevatedButton.styleFrom(
          backgroundColor: Colors.orange,
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
        ),
        child: const Text('활동량 & 비만도 보기 →',
            style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold)),
      ),
    );
  }
}