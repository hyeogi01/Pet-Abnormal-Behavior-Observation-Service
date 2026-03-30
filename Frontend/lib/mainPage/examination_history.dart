import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

class ExaminationHistoryPage extends StatefulWidget {
  final Map<String, dynamic>? petData;
  final String userId;

  const ExaminationHistoryPage({super.key, required this.userId, this.petData});

  @override
  State<ExaminationHistoryPage> createState() => _ExaminationHistoryPageState();
}

class _ExaminationHistoryPageState extends State<ExaminationHistoryPage> {
  bool _isLoading = true;
  List<dynamic> _history = [];

  @override
  void initState() {
    super.initState();
    _fetchHistory();
  }

  Future<void> _fetchHistory() async {
    try {
      final response = await http.get(Uri.parse('http://localhost:8080/api/examination-history/${widget.userId}'));
      if (response.statusCode == 200) {
        final decoded = jsonDecode(utf8.decode(response.bodyBytes));
        if (decoded['status'] == 'success') {
          setState(() {
            _history = decoded['data'] ?? [];
            _isLoading = false;
          });
          return;
        }
      }
    } catch (e) {
      print('History fetch error: $e');
    }
    setState(() => _isLoading = false);
  }

  @override
  Widget build(BuildContext context) {
    String petName = widget.petData?['pet_name'] ?? '반려동물';
    
    return Scaffold(
      backgroundColor: Colors.grey[50],
      appBar: AppBar(
        title: Text('$petName의 검진 기록', style: const TextStyle(fontWeight: FontWeight.bold)),
        backgroundColor: Colors.white,
        foregroundColor: Colors.black,
        elevation: 0,
      ),
      body: _isLoading 
        ? const Center(child: CircularProgressIndicator(color: Colors.green))
        : _history.isEmpty
          ? const Center(child: Text('아직 검진 기록이 없습니다.', style: TextStyle(color: Colors.grey)))
          : ListView.builder(
              padding: const EdgeInsets.all(16),
              itemCount: _history.length,
              itemBuilder: (context, index) {
                final item = _history[index];
                bool isEye = item['category'] == 'eye';
                final result = item['result'] ?? {};
                String title = isEye ? '👁️ 안구 검사' : '🩹 피부 검사';
                
                // Result has fields depending on AI engine, user expects diagnosis and probability
                String diagnosis = result['diagnosis'] ?? '정상';
                String prob = result['probability'] != null ? '${double.parse(result['probability'].toString()).toStringAsFixed(1)}%' : '';
                String imageUrl = item['image_url'] ?? '';

                // Replace minio internal host if necessary
                if (imageUrl.contains('minio:9000')) {
                  imageUrl = imageUrl.replaceAll('minio:9000', 'localhost:9000');
                }

                return Card(
                  margin: const EdgeInsets.only(bottom: 16),
                  shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
                  elevation: 2,
                  child: Padding(
                    padding: const EdgeInsets.all(16.0),
                    child: Row(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        // Image Thumbnail
                        if (imageUrl.isNotEmpty)
                          ClipRRect(
                            borderRadius: BorderRadius.circular(12),
                            child: Image.network(
                              imageUrl,
                              width: 80,
                              height: 80,
                              fit: BoxFit.cover,
                              errorBuilder: (context, error, stackTrace) => 
                                Container(
                                  width: 80, height: 80, color: Colors.grey[200],
                                  child: Icon(isEye ? Icons.visibility : Icons.healing, color: Colors.grey[400]),
                                )
                            ),
                          )
                        else
                          Container(
                            width: 80, height: 80, decoration: BoxDecoration(color: Colors.grey[200], borderRadius: BorderRadius.circular(12)),
                            child: Icon(isEye ? Icons.visibility : Icons.healing, color: Colors.grey[400]),
                          ),
                        const SizedBox(width: 16),
                        // Details
                        Expanded(
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Row(
                                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                                children: [
                                  Text(title, style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 16)),
                                  Text('${item['date'] ?? ''}', style: const TextStyle(color: Colors.grey, fontSize: 12)),
                                ],
                              ),
                              const SizedBox(height: 8),
                              Container(
                                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                                decoration: BoxDecoration(
                                  color: diagnosis.contains('정상') ? Colors.green[50] : Colors.orange[50],
                                  borderRadius: BorderRadius.circular(6)
                                ),
                                child: Text(
                                  diagnosis, 
                                  style: TextStyle(
                                    color: diagnosis.contains('정상') ? Colors.green[700] : Colors.deepOrange, 
                                    fontSize: 13, 
                                    fontWeight: FontWeight.bold
                                  )
                                ),
                              ),
                              const SizedBox(height: 6),
                              if (prob.isNotEmpty)
                                Text('AI 확신도: $prob', style: const TextStyle(color: Colors.black54, fontSize: 12)),
                            ],
                          ),
                        )
                      ],
                    ),
                  ),
                );
              },
            ),
    );
  }
}
