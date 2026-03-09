import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

class PhotoGalleryPage extends StatefulWidget {
  final String userId;
  const PhotoGalleryPage({super.key, required this.userId});

  @override
  State<PhotoGalleryPage> createState() => _PhotoGalleryPageState();
}

class _PhotoGalleryPageState extends State<PhotoGalleryPage> {
  final String baseUrl = 'http://localhost:8080'; // Make sure to change to 10.0.2.2 or real IP
  List<dynamic> galleryItems = [];
  bool isLoading = true;

  @override
  void initState() {
    super.initState();
    _fetchGalleryData();
  }

  Future<void> _fetchGalleryData() async {
    final url = Uri.parse('$baseUrl/api/gallery/${widget.userId}');
    try {
      final response = await http.get(url);
      if (response.statusCode == 200) {
        final decoded = jsonDecode(response.body);
        if (decoded['status'] == 'success') {
          setState(() {
            galleryItems = decoded['data'] ?? [];
            isLoading = false;
          });
        }
      } else {
        setState(() => isLoading = false);
      }
    } catch (e) {
      print('Gallery fetch error: $e');
      setState(() => isLoading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.white,
      appBar: AppBar(
        title: const Text('활동 사진첩', style: TextStyle(color: Colors.black, fontWeight: FontWeight.bold, fontSize: 16)),
        backgroundColor: Colors.white,
        elevation: 0,
        centerTitle: true,
      ),
      body: isLoading
          ? const Center(child: CircularProgressIndicator())
          : galleryItems.isEmpty
              ? const Center(child: Text('AI가 모아둔 비디오/사진 기록이 아직 없습니다.'))
              : GridView.builder(
                  padding: const EdgeInsets.all(8),
                  gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
                    crossAxisCount: 3, // 3 items per row like mobile galleries
                    crossAxisSpacing: 4,
                    mainAxisSpacing: 4,
                  ),
                  itemCount: galleryItems.length,
                  itemBuilder: (context, index) {
                    final item = galleryItems[index];
                    return _buildGalleryThumbnail(item);
                  },
                ),
    );
  }

  Widget _buildGalleryThumbnail(Map<String, dynamic> item) {
    final emotion = item['emotion'] ?? 'Unknown';
    final timestamp = item['timestamp'] ?? '';
    final videoUrl = item['video_url'] ?? '';

    // Choose border or icon color based on emotion
    Color emotionColor = Colors.grey;
    if (emotion.contains('happy') || emotion.contains('relaxed')) {
      emotionColor = Colors.green;
    } else if (emotion.contains('anxious') || emotion.contains('scared') || emotion.contains('stress')) {
      emotionColor = Colors.red;
    }

    return GestureDetector(
      onTap: () {
        // Here you could navigate to a Video Player screen
        _showVideoDialog(videoUrl, timestamp, emotionColor);
      },
      child: Stack(
        fit: StackFit.expand,
        children: [
          // Display the actual extracted image frame from MinIO via URL
          Container(
            color: Colors.grey[300],
            child: videoUrl.isNotEmpty 
              ? Image.network(
                  videoUrl,
                  fit: BoxFit.cover,
                  errorBuilder: (context, error, stackTrace) {
                    return const Icon(Icons.image_not_supported, color: Colors.white, size: 40);
                  },
                )
              : const Icon(Icons.play_circle_outline, color: Colors.white, size: 40),
          ),
          // Decorating Emotion Status
          Positioned(
            top: 4,
            right: 4,
            child: CircleAvatar(
              radius: 6,
              backgroundColor: emotionColor,
            ),
          ),
        ],
      ),
    );
  }

  void _showVideoDialog(String url, String time, Color themeColor) {
    showDialog(
      context: context,
      builder: (context) {
        return AlertDialog(
          title: Text(time, style: const TextStyle(fontSize: 14)),
          content: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              const Text("비디오 재생 기능은 추후 연동됩니다."),
              const SizedBox(height: 10),
              Text(url, style: const TextStyle(fontSize: 10, color: Colors.grey)),
            ],
          ),
          actions: [
            TextButton(onPressed: () => Navigator.pop(context), child: const Text("닫기"))
          ],
        );
      }
    );
  }
}
