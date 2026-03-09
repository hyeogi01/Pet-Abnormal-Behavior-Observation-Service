import 'package:flutter/material.dart';

class DiaryDetailPage extends StatelessWidget {
  final Map<String, dynamic> diaryData;

  const DiaryDetailPage({super.key, required this.diaryData});

  @override
  Widget build(BuildContext context) {
    final date = diaryData['date'] ?? '알 수 없는 날짜';
    final content = diaryData['content'] ?? '일기 내용 없음';
    final petType = diaryData['pet_type'] ?? '반려동물';

    return Scaffold(
      backgroundColor: Colors.white,
      appBar: AppBar(
        title: const Text('AI 행동 관찰 일기', style: TextStyle(color: Colors.black, fontSize: 16, fontWeight: FontWeight.bold)),
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
            // Header Image Placeholder (Can be linked to photo gallery thumbnail later)
            Container(
              width: double.infinity,
              height: 200,
              decoration: BoxDecoration(
                color: Colors.grey[200],
                borderRadius: BorderRadius.circular(15),
              ),
              child: ClipRRect(
                borderRadius: BorderRadius.circular(15),
                child: diaryData['video_url'] != null && diaryData['video_url'] != ''
                    ? Image.network(
                        diaryData['video_url'],
                        fit: BoxFit.cover,
                        errorBuilder: (context, error, stackTrace) =>
                            const Center(child: Icon(Icons.pets, size: 50, color: Colors.grey)),
                      )
                    : const Center(child: Icon(Icons.pets, size: 50, color: Colors.grey)),
              ),
            ),
            const SizedBox(height: 20),
            
            // Date Title
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Text(
                  date,
                  style: const TextStyle(fontSize: 22, fontWeight: FontWeight.bold, color: Colors.black87),
                ),
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                  decoration: BoxDecoration(
                    color: Colors.blue[50],
                    borderRadius: BorderRadius.circular(20),
                  ),
                  child: Text(
                    petType,
                    style: TextStyle(color: Colors.blue[800], fontSize: 12, fontWeight: FontWeight.bold),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 15),
            const Divider(),
            const SizedBox(height: 15),
            
            // Diary Content
            Text(
              content,
              style: const TextStyle(
                fontSize: 15,
                height: 1.8,  // Line height for readability
                color: Colors.black87,
              ),
            ),
          ],
        ),
      ),
    );
  }
}
