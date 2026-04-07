import 'package:flutter/material.dart';
import 'package:fl_chart/fl_chart.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

class WeightHistoryPage extends StatefulWidget {
  final String userId;
  const WeightHistoryPage({super.key, required this.userId});

  @override
  State<WeightHistoryPage> createState() => _WeightHistoryPageState();
}

class _WeightHistoryPageState extends State<WeightHistoryPage> {
  List<Map<String, dynamic>> _weightData = [];
  bool _isLoading = true;
  final String baseUrl = "http://localhost:8080";

  @override
  void initState() {
    super.initState();
    _fetchWeights();
  }

  Future<void> _fetchWeights() async {
    try {
      final url = Uri.parse('$baseUrl/api/weight/${widget.userId}');
      final response = await http.get(url);
      if (response.statusCode == 200) {
        final decoded = jsonDecode(response.body);
        if (decoded['status'] == 'success') {
          setState(() {
            _weightData = List<Map<String, dynamic>>.from(decoded['data']);
            // 포맷 변환 (2026-04-06 -> 04.06)
            for (var item in _weightData) {
              if (item['date'] != null && item['date'].contains('-')) {
                List<String> parts = item['date'].split('-');
                if (parts.length >= 3) {
                  item['short_date'] = '${parts[1]}.${parts[2]}';
                }
              }
            }
          });
        }
      }
    } catch (e) {
      print('Weight fetch error: $e');
    } finally {
      setState(() => _isLoading = false);
    }
  }

  void _showAddWeightDialog() {
    final TextEditingController weightController = TextEditingController();
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
        title: const Text('몸무게 기록', style: TextStyle(fontWeight: FontWeight.bold)),
        content: TextField(
          controller: weightController,
          keyboardType: const TextInputType.numberWithOptions(decimal: true),
          decoration: InputDecoration(
            hintText: 'kg 단위로 입력',
            suffixText: 'kg',
            filled: true,
            fillColor: Colors.grey[100],
            border: OutlineInputBorder(borderRadius: BorderRadius.circular(12), borderSide: BorderSide.none),
          ),
        ),
        actionsPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
        actions: [
          TextButton(onPressed: () => Navigator.pop(context), child: const Text('취소', style: TextStyle(color: Colors.grey))),
          ElevatedButton(
            onPressed: () async {
              if (weightController.text.isNotEmpty) {
                final double? weightValue = double.tryParse(weightController.text);
                if (weightValue == null) return;
                
                Navigator.pop(context); // close dialog immediately for responsive UX
                
                // 오늘 날짜
                final now = DateTime.now();
                final String todayStr = "${now.year}-${now.month.toString().padLeft(2, '0')}-${now.day.toString().padLeft(2, '0')}";

                // 서버 전송
                try {
                  final url = Uri.parse('$baseUrl/api/weight/${widget.userId}');
                  await http.post(
                    url,
                    headers: {'Content-Type': 'application/json'},
                    body: jsonEncode({"date": todayStr, "weight": weightValue}),
                  );
                  // 새로고침
                  _fetchWeights();
                } catch(e) {
                  print("Weight save error: $e");
                }
              }
            },
            style: ElevatedButton.styleFrom(elevation: 0, backgroundColor: Colors.blueAccent),
            child: const Text('저장', style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold)),
          ),
        ],
      )
    );
  }

  @override
  Widget build(BuildContext context) {
    // 가장 최근 10개만 차트에 표시용
    List<Map<String, dynamic>> chartData = _weightData.length > 10 ? _weightData.sublist(_weightData.length - 10) : _weightData;

    return Scaffold(
      backgroundColor: Colors.white,
      appBar: AppBar(
        title: const Text('몸무게 기록', style: TextStyle(color: Colors.black, fontWeight: FontWeight.bold)),
        backgroundColor: Colors.white,
        iconTheme: const IconThemeData(color: Colors.black),
        elevation: 0,
      ),
      body: _isLoading 
        ? const Center(child: CircularProgressIndicator())
        : Padding(
            padding: const EdgeInsets.all(16.0),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                const Text('최근 체중 변화 추이', style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold, color: Colors.blueGrey)),
                const SizedBox(height: 20),
                SizedBox(
                  height: 250,
                  child: chartData.isEmpty ? const Center(child: Text('기록이 없습니다.', style: TextStyle(color: Colors.grey))) : LineChart(
                    LineChartData(
                      gridData: const FlGridData(show: true, drawVerticalLine: false),
                      titlesData: FlTitlesData(
                        show: true,
                        topTitles: const AxisTitles(sideTitles: SideTitles(showTitles: false)),
                        rightTitles: const AxisTitles(sideTitles: SideTitles(showTitles: false)),
                        bottomTitles: AxisTitles(
                          sideTitles: SideTitles(
                            showTitles: true,
                            getTitlesWidget: (value, meta) {
                              final index = value.toInt();
                              if (index >= 0 && index < chartData.length) {
                                return Padding(
                                  padding: const EdgeInsets.only(top: 8.0),
                                  child: Text(chartData[index]['short_date'] ?? '', style: const TextStyle(fontSize: 11, color: Colors.grey)),
                                );
                              }
                              return const Text('');
                            },
                          ),
                        ),
                      ),
                      borderData: FlBorderData(show: false),
                      lineBarsData: [
                        LineChartBarData(
                          spots: chartData.asMap().entries.map((e) => FlSpot(e.key.toDouble(), (e.value['weight'] as num).toDouble())).toList(),
                          isCurved: true,
                          color: Colors.blueAccent,
                          barWidth: 4,
                          isStrokeCapRound: true,
                          dotData: FlDotData(show: true, getDotPainter: (spot, percent, barData, index) {
                            return FlDotCirclePainter(radius: 4, color: Colors.white, strokeWidth: 2, strokeColor: Colors.blueAccent);
                          }),
                          belowBarData: BarAreaData(
                            show: true, 
                            gradient: LinearGradient(
                              colors: [Colors.blueAccent.withOpacity(0.3), Colors.white.withOpacity(0.0)],
                              begin: Alignment.topCenter,
                              end: Alignment.bottomCenter,
                            )
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
                const SizedBox(height: 30),
                Expanded(
                  child: ListView.builder(
                    itemCount: _weightData.length,
                    itemBuilder: (context, index) {
                      final data = _weightData.reversed.toList()[index];
                      return Container(
                        margin: const EdgeInsets.only(bottom: 10),
                        decoration: BoxDecoration(
                          color: Colors.white,
                          borderRadius: BorderRadius.circular(15),
                          boxShadow: [BoxShadow(color: Colors.black.withOpacity(0.05), blurRadius: 10, offset: const Offset(0, 4))],
                        ),
                        child: ListTile(
                          contentPadding: const EdgeInsets.symmetric(horizontal: 20, vertical: 4),
                          leading: Container(
                            padding: const EdgeInsets.all(8),
                            decoration: BoxDecoration(color: Colors.blueAccent.withOpacity(0.1), shape: BoxShape.circle),
                            child: const Icon(Icons.monitor_weight, color: Colors.blueAccent, size: 20)
                          ),
                          title: Text('${data['date']}', style: const TextStyle(color: Colors.grey, fontSize: 13)),
                          trailing: Text('${data['weight']} kg', style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 18, color: Colors.black87)),
                        ),
                      );
                    },
                  ),
                ),
              ],
            ),
          ),
      floatingActionButton: FloatingActionButton(
        onPressed: _showAddWeightDialog,
        backgroundColor: Colors.blueAccent,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(15)),
        child: const Icon(Icons.add, color: Colors.white),
      ),
    );
  }
}
