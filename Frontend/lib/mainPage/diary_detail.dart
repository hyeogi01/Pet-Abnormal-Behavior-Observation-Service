import 'package:flutter/material.dart';

class DiaryDetailPage extends StatelessWidget {
  final Map<String, dynamic> diaryData;

  const DiaryDetailPage({super.key, required this.diaryData});

  @override
  Widget build(BuildContext context) {
    final date = diaryData['date'] ?? '알 수 없는 날짜';
    final petDiary = diaryData['pet_diary'] ?? diaryData['content'] ?? '일기가 아직 작성되지 않았습니다.';
    final report = diaryData['report'] ?? '분석 레포트가 없습니다.';
    final memo = diaryData['memo'] ?? '';
    final petType = diaryData['pet_type'] ?? '반려동물';
    final List<String> imageUrls = List<String>.from(diaryData['image_urls'] ?? []);

    return Scaffold(
      backgroundColor: const Color(0xFFF9F9F9),
      appBar: AppBar(
        title: Text('$date 일기', style: const TextStyle(color: Colors.black, fontSize: 16, fontWeight: FontWeight.bold)),
        backgroundColor: Colors.white,
        elevation: 0,
        leading: IconButton(
          icon: const Icon(Icons.arrow_back_ios, color: Colors.black, size: 20),
          onPressed: () => Navigator.pop(context),
        ),
        centerTitle: true,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // 📸 Photo Gallery (Main + 4 subs)
            _buildPhotoGallery(imageUrls),
            const SizedBox(height: 24),
            
            // 💬 Pet Diary
            _buildSectionTitle('🐾 반려동물의 시점'),
            _buildContentCard(petDiary, Colors.blue.shade50, Colors.blueGrey),
            const SizedBox(height: 24),

            // 📋 AI Behavior Report
            _buildSectionTitle('📋 AI 행동 분석 레포트'),
            _buildReportCard(report),
            const SizedBox(height: 24),

            // ✍️ Protector's Memo
            if (memo.isNotEmpty) ...[
              _buildSectionTitle('✍️ 보호자 메모'),
              _buildContentCard(memo, Colors.orange.shade50, Colors.orange.shade900),
              const SizedBox(height: 24),
            ],
          ],
        ),
      ),
    );
  }

  Widget _buildSectionTitle(String title) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 12.0),
      child: Text(title, style: const TextStyle(fontSize: 15, fontWeight: FontWeight.bold, color: Colors.black87)),
    );
  }

  Widget _buildContentCard(String content, Color bgColor, Color textColor) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(16.0),
      decoration: BoxDecoration(color: bgColor, borderRadius: BorderRadius.circular(12)),
      child: Text(content, style: TextStyle(color: textColor, fontSize: 13, height: 1.5)),
    );
  }

  Widget _buildReportCard(String report) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(16.0),
      decoration: BoxDecoration(
        gradient: LinearGradient(colors: [Colors.blue.shade400, Colors.indigo.shade400]),
        borderRadius: BorderRadius.circular(15),
      ),
      child: Text(
        report,
        style: const TextStyle(color: Colors.white, fontSize: 13, height: 1.5, fontWeight: FontWeight.w500),
      ),
    );
  }

  Widget _buildPhotoGallery(List<String> images) {
    return Column(
      children: [
        Container(
          height: 175,
          width: double.infinity,
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(15),
            color: Colors.grey[200],
            image: images.isNotEmpty
                ? DecorationImage(image: NetworkImage(images[0]), fit: BoxFit.cover)
                : const DecorationImage(image: NetworkImage('https://via.placeholder.com/600x400'), fit: BoxFit.cover),
          ),
        ),
        const SizedBox(height: 8),
        if (images.length > 1)
          GridView.builder(
            shrinkWrap: true,
            physics: const NeverScrollableScrollPhysics(),
            itemCount: 4,
            gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
              crossAxisCount: 4,
              crossAxisSpacing: 8,
              mainAxisSpacing: 8,
              childAspectRatio: 1,
            ),
            itemBuilder: (context, index) {
              final imgIndex = index + 1;
              return Container(
                decoration: BoxDecoration(
                  borderRadius: BorderRadius.circular(10),
                  color: Colors.grey[200],
                  image: images.length > imgIndex
                      ? DecorationImage(image: NetworkImage(images[imgIndex]), fit: BoxFit.cover)
                      : const DecorationImage(image: NetworkImage('https://via.placeholder.com/150'), fit: BoxFit.cover),
                ),
              );
            },
          ),
      ],
    );
  }
}
