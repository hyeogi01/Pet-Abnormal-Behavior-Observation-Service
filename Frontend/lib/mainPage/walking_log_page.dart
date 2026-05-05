import 'package:flutter/material.dart';
import 'package:flutter/cupertino.dart';
import 'package:fl_chart/fl_chart.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

class WalkingLogPage extends StatefulWidget {
  final String userId;
  const WalkingLogPage({super.key, required this.userId});

  @override
  State<WalkingLogPage> createState() => _WalkingLogPageState();
}

class _WalkingLogPageState extends State<WalkingLogPage> {
  List<Map<String, dynamic>> _walkingLogs = [];
  bool _isLoading = true;
  final String baseUrl = "http://localhost:8080";

  @override
  void initState() {
    super.initState();
    _fetchWalkingLogs();
  }

  Future<void> _fetchWalkingLogs() async {
    try {
      final url = Uri.parse('$baseUrl/api/walking-logs/${widget.userId}');
      final response = await http.get(url);
      if (response.statusCode == 200) {
        final decoded = jsonDecode(response.body);
        if (decoded['status'] == 'success') {
          setState(() {
            _walkingLogs = List<Map<String, dynamic>>.from(decoded['data']);
            _isLoading = false;
          });
          return;
        }
      }
    } catch (e) {
      print('Walking log fetch error: $e');
    }
    setState(() => _isLoading = false);
  }

  /// 날짜별로 로그를 그룹화
  Map<String, List<Map<String, dynamic>>> _groupByDate() {
    final Map<String, List<Map<String, dynamic>>> grouped = {};
    for (var log in _walkingLogs) {
      final date = log['date'] ?? '';
      grouped.putIfAbsent(date, () => []);
      grouped[date]!.add(log);
    }
    // 날짜 역순 정렬을 위해 LinkedHashMap 대신 정렬된 키를 사용
    final sortedKeys = grouped.keys.toList()..sort((a, b) => b.compareTo(a));
    final sortedMap = <String, List<Map<String, dynamic>>>{};
    for (var key in sortedKeys) {
      // 각 날짜 내 시간순 정렬
      grouped[key]!.sort((a, b) => (a['start_time'] ?? '').compareTo(b['start_time'] ?? ''));
      sortedMap[key] = grouped[key]!;
    }
    return sortedMap;
  }

  /// 날짜별 산책 시간 합산 (중복 없는 차트 데이터)
  List<Map<String, dynamic>> _getChartData() {
    final Map<String, int> dateMap = {};
    for (var log in _walkingLogs) {
      final date = log['date'] ?? '';
      final duration = (log['duration_min'] as num?)?.toInt() ?? 0;
      dateMap[date] = (dateMap[date] ?? 0) + duration;
    }

    final sorted = dateMap.entries.toList()
      ..sort((a, b) => a.key.compareTo(b.key));

    // 최근 7일 데이터만
    final recent = sorted.length > 7 ? sorted.sublist(sorted.length - 7) : sorted;
    return recent.map((e) => {"date": e.key, "duration": e.value}).toList();
  }

  String _formatDateShort(String dateStr) {
    if (dateStr.contains('-')) {
      final parts = dateStr.split('-');
      if (parts.length >= 3) return '${parts[1]}.${parts[2]}';
    }
    return dateStr;
  }

  Future<void> _deleteLog(String logId) async {
    try {
      final url = Uri.parse('$baseUrl/api/walking-logs/${widget.userId}/$logId');
      final response = await http.delete(url);
      if (response.statusCode == 200) {
        final decoded = jsonDecode(response.body);
        if (decoded['status'] == 'success') {
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(content: Text('산책 기록이 삭제되었습니다.')),
          );
          _fetchWalkingLogs(); // 새로고침
          return;
        }
      }
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('삭제에 실패했습니다.')),
      );
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('서버 통신 에러: $e')),
      );
    }
  }

  Future<void> _updateLog(String logId, String date, String startTime, String endTime) async {
    try {
      final url = Uri.parse('$baseUrl/api/walking-logs/${widget.userId}/$logId');
      final response = await http.put(
        url,
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
          'date': date,
          'start_time': startTime,
          'end_time': endTime,
        }),
      );
      if (response.statusCode == 200) {
        final decoded = jsonDecode(response.body);
        if (decoded['status'] == 'success') {
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(content: Text('산책 기록이 수정되었습니다.')),
          );
          _fetchWalkingLogs(); // 새로고침
          return;
        }
      }
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('수정에 실패했습니다.')),
      );
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('서버 통신 에러: $e')),
      );
    }
  }

  void _showSessionManageSheet(String date, List<Map<String, dynamic>> sessions) {
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (context) => Container(
        decoration: const BoxDecoration(
          color: Color(0xFF2C2C2E),
          borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
        ),
        padding: const EdgeInsets.all(20),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Center(
              child: Container(
                width: 40, height: 4,
                decoration: BoxDecoration(
                  color: Colors.grey[600],
                  borderRadius: BorderRadius.circular(2),
                ),
              ),
            ),
            const SizedBox(height: 16),
            Text(
              '$date 산책 기록',
              style: const TextStyle(color: Colors.white, fontSize: 18, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 4),
            Text(
              '수정하거나 삭제할 기록을 선택하세요',
              style: TextStyle(color: Colors.grey[400], fontSize: 13),
            ),
            const SizedBox(height: 16),
            ...sessions.map((session) {
              final startTime = session['start_time'] ?? '';
              final endTime = session['end_time'] ?? '';
              final duration = (session['duration_min'] as num?)?.toInt() ?? 0;
              final logId = session['id'] ?? '';

              return Container(
                margin: const EdgeInsets.only(bottom: 10),
                padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
                decoration: BoxDecoration(
                  color: Colors.grey[800],
                  borderRadius: BorderRadius.circular(12),
                ),
                child: Row(
                  children: [
                    const Icon(Icons.access_time, color: Colors.white70, size: 18),
                    const SizedBox(width: 10),
                    Expanded(
                      child: Text(
                        '$startTime ~ $endTime (${duration}분)',
                        style: const TextStyle(color: Colors.white, fontSize: 14),
                      ),
                    ),
                    // 수정 버튼
                    IconButton(
                      icon: const Icon(Icons.edit, color: Colors.orange, size: 20),
                      onPressed: () {
                        Navigator.pop(context);
                        _showEditDialog(logId, date, startTime, endTime);
                      },
                      tooltip: '수정',
                      constraints: const BoxConstraints(),
                      padding: const EdgeInsets.all(8),
                    ),
                    // 삭제 버튼
                    IconButton(
                      icon: const Icon(Icons.delete_outline, color: Colors.redAccent, size: 20),
                      onPressed: () {
                        Navigator.pop(context);
                        _showDeleteConfirmDialog(logId);
                      },
                      tooltip: '삭제',
                      constraints: const BoxConstraints(),
                      padding: const EdgeInsets.all(8),
                    ),
                  ],
                ),
              );
            }),
            const SizedBox(height: 8),
          ],
        ),
      ),
    );
  }

  void _showDeleteConfirmDialog(String logId) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('산책 기록 삭제'),
        content: const Text('이 기록을 삭제하시겠습니까?'),
        actions: [
          TextButton(onPressed: () => Navigator.pop(context), child: const Text('취소')),
          TextButton(
            onPressed: () {
              Navigator.pop(context);
              _deleteLog(logId);
            },
            child: const Text('삭제', style: TextStyle(color: Colors.red)),
          ),
        ],
      ),
    );
  }

  void _showEditDialog(String logId, String date, String currentStart, String currentEnd) {
    // 시작/종료 시간 파싱
    final startParts = currentStart.split(':');
    final endParts = currentEnd.split(':');
    TimeOfDay startTime = TimeOfDay(
      hour: int.tryParse(startParts[0]) ?? 9,
      minute: int.tryParse(startParts[1]) ?? 0,
    );
    TimeOfDay endTime = TimeOfDay(
      hour: int.tryParse(endParts[0]) ?? 10,
      minute: int.tryParse(endParts[1]) ?? 0,
    );

    String formatTime(TimeOfDay t) => '${t.hour.toString().padLeft(2, '0')}:${t.minute.toString().padLeft(2, '0')}';

    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (context) => StatefulBuilder(
        builder: (context, setModalState) {
          return Container(
            decoration: const BoxDecoration(
              color: Color(0xFF2C2C2E),
              borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
            ),
            padding: const EdgeInsets.all(20),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                Container(
                  width: 40, height: 4,
                  decoration: BoxDecoration(
                    color: Colors.grey[600],
                    borderRadius: BorderRadius.circular(2),
                  ),
                ),
                const SizedBox(height: 16),
                const Text(
                  '산책 기록 수정',
                  style: TextStyle(color: Colors.white, fontSize: 18, fontWeight: FontWeight.bold),
                ),
                const SizedBox(height: 4),
                Text(
                  '$date',
                  style: TextStyle(color: Colors.grey[400], fontSize: 13),
                ),
                const SizedBox(height: 20),

                // 시간 선택 (시작 / 종료)
                Row(
                  children: [
                    Expanded(
                      child: GestureDetector(
                        onTap: () {
                          _showCupertinoTimePicker(
                            context,
                            initialTime: startTime,
                            onTimeChanged: (picked) {
                              setModalState(() => startTime = picked);
                            },
                          );
                        },
                        child: Container(
                          padding: const EdgeInsets.symmetric(vertical: 14),
                          decoration: BoxDecoration(
                            color: Colors.grey[800],
                            borderRadius: BorderRadius.circular(12),
                          ),
                          child: Column(
                            children: [
                              Text('시작', style: TextStyle(color: Colors.grey[400], fontSize: 12)),
                              const SizedBox(height: 6),
                              Text(
                                formatTime(startTime),
                                style: const TextStyle(color: Colors.white, fontSize: 24, fontWeight: FontWeight.bold),
                              ),
                            ],
                          ),
                        ),
                      ),
                    ),
                    Padding(
                      padding: const EdgeInsets.symmetric(horizontal: 12),
                      child: Icon(Icons.arrow_forward, color: Colors.grey[500]),
                    ),
                    Expanded(
                      child: GestureDetector(
                        onTap: () {
                          _showCupertinoTimePicker(
                            context,
                            initialTime: endTime,
                            onTimeChanged: (picked) {
                              setModalState(() => endTime = picked);
                            },
                          );
                        },
                        child: Container(
                          padding: const EdgeInsets.symmetric(vertical: 14),
                          decoration: BoxDecoration(
                            color: Colors.grey[800],
                            borderRadius: BorderRadius.circular(12),
                          ),
                          child: Column(
                            children: [
                              Text('종료', style: TextStyle(color: Colors.grey[400], fontSize: 12)),
                              const SizedBox(height: 6),
                              Text(
                                formatTime(endTime),
                                style: const TextStyle(color: Colors.white, fontSize: 24, fontWeight: FontWeight.bold),
                              ),
                            ],
                          ),
                        ),
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 20),

                SizedBox(
                  width: double.infinity,
                  child: ElevatedButton(
                    onPressed: () {
                      if (endTime.hour < startTime.hour || (endTime.hour == startTime.hour && endTime.minute <= startTime.minute)) {
                        ScaffoldMessenger.of(context).showSnackBar(
                          const SnackBar(content: Text('종료 시간은 시작 시간 이후여야 합니다.')),
                        );
                        return;
                      }
                      Navigator.pop(context);
                      _updateLog(logId, date, formatTime(startTime), formatTime(endTime));
                    },
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Colors.orange,
                      foregroundColor: Colors.white,
                      elevation: 0,
                      padding: const EdgeInsets.symmetric(vertical: 14),
                      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                    ),
                    child: const Text('수정 완료', style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold)),
                  ),
                ),
                const SizedBox(height: 16),
              ],
            ),
          );
        },
      ),
    );
  }

  void _showCupertinoTimePicker(BuildContext context, {required TimeOfDay initialTime, required Function(TimeOfDay) onTimeChanged}) {
    DateTime tempDateTime = DateTime(2026, 1, 1, initialTime.hour, initialTime.minute);

    showModalBottomSheet(
      context: context,
      backgroundColor: Colors.transparent,
      builder: (ctx) => Container(
        height: 300,
        decoration: const BoxDecoration(
          color: Color(0xFF2C2C2E),
          borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
        ),
        child: Column(
          children: [
            const SizedBox(height: 8),
            Container(
              width: 40, height: 4,
              decoration: BoxDecoration(
                color: Colors.grey[600],
                borderRadius: BorderRadius.circular(2),
              ),
            ),
            Expanded(
              child: CupertinoTheme(
                data: const CupertinoThemeData(
                  brightness: Brightness.dark,
                  textTheme: CupertinoTextThemeData(
                    dateTimePickerTextStyle: TextStyle(color: Colors.white, fontSize: 22),
                  ),
                ),
                child: CupertinoDatePicker(
                  mode: CupertinoDatePickerMode.time,
                  initialDateTime: tempDateTime,
                  use24hFormat: false,
                  onDateTimeChanged: (DateTime newTime) {
                    tempDateTime = newTime;
                  },
                ),
              ),
            ),
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
              child: SizedBox(
                width: double.infinity,
                child: ElevatedButton(
                  onPressed: () {
                    onTimeChanged(TimeOfDay(hour: tempDateTime.hour, minute: tempDateTime.minute));
                    Navigator.pop(ctx);
                  },
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.orange,
                    foregroundColor: Colors.white,
                    elevation: 0,
                    padding: const EdgeInsets.symmetric(vertical: 14),
                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                  ),
                  child: const Text('확인', style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold)),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final grouped = _groupByDate();

    return Scaffold(
      backgroundColor: Colors.grey[50],
      appBar: AppBar(
        title: const Text('산책 로그', style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16)),
        backgroundColor: Colors.white,
        foregroundColor: Colors.black,
        elevation: 0.5,
      ),
      body: _isLoading
          ? const Center(child: CircularProgressIndicator())
          : _walkingLogs.isEmpty
              ? _buildEmptyState()
              : SingleChildScrollView(
                  padding: const EdgeInsets.all(16),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      _buildStatisticsChart(),
                      const SizedBox(height: 24),
                      const Text(
                        '산책 내역',
                        style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                      ),
                      const SizedBox(height: 12),
                      ...grouped.entries.map((entry) => _buildGroupedLogItem(entry.key, entry.value)),
                    ],
                  ),
                ),
    );
  }

  Widget _buildEmptyState() {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(Icons.pets, size: 64, color: Colors.grey[300]),
          const SizedBox(height: 16),
          Text(
            '아직 산책 기록이 없어요',
            style: TextStyle(color: Colors.grey[500], fontSize: 16),
          ),
          const SizedBox(height: 8),
          Text(
            '마이페이지에서 산책 시간을 설정해 보세요!',
            style: TextStyle(color: Colors.grey[400], fontSize: 13),
          ),
        ],
      ),
    );
  }

  Widget _buildStatisticsChart() {
    final chartData = _getChartData();
    if (chartData.isEmpty) return const SizedBox.shrink();

    final maxDuration = chartData.map((e) => e['duration'] as int).reduce((a, b) => a > b ? a : b);

    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(16),
        boxShadow: const [BoxShadow(color: Colors.black12, blurRadius: 6)],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(Icons.bar_chart_rounded, color: Colors.blue[600], size: 22),
              const SizedBox(width: 8),
              const Text(
                '산책 통계',
                style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
              ),
            ],
          ),
          const SizedBox(height: 6),
          Text(
            '최근 ${chartData.length}일간의 산책 시간(분)',
            style: TextStyle(fontSize: 12, color: Colors.grey[500]),
          ),
          const SizedBox(height: 20),
          SizedBox(
            height: 200,
            child: LineChart(
              LineChartData(
                gridData: FlGridData(
                  show: true,
                  drawVerticalLine: false,
                  horizontalInterval: (maxDuration / 4).ceilToDouble().clamp(1, double.infinity),
                  getDrawingHorizontalLine: (value) => FlLine(
                    color: Colors.grey[200]!,
                    strokeWidth: 1,
                  ),
                ),
                titlesData: FlTitlesData(
                  leftTitles: AxisTitles(
                    sideTitles: SideTitles(
                      showTitles: true,
                      reservedSize: 32,
                      getTitlesWidget: (value, meta) => Text(
                        '${value.toInt()}',
                        style: TextStyle(fontSize: 10, color: Colors.grey[500]),
                      ),
                    ),
                  ),
                  bottomTitles: AxisTitles(
                    sideTitles: SideTitles(
                      showTitles: true,
                      reservedSize: 28,
                      interval: 1,
                      getTitlesWidget: (value, meta) {
                        final idx = value.toInt();
                        if (value != idx.toDouble()) return const SizedBox.shrink();
                        if (idx >= 0 && idx < chartData.length) {
                          return Padding(
                            padding: const EdgeInsets.only(top: 8),
                            child: Text(
                              _formatDateShort(chartData[idx]['date']),
                              style: TextStyle(fontSize: 10, color: Colors.grey[600]),
                            ),
                          );
                        }
                        return const SizedBox.shrink();
                      },
                    ),
                  ),
                  rightTitles: const AxisTitles(sideTitles: SideTitles(showTitles: false)),
                  topTitles: const AxisTitles(sideTitles: SideTitles(showTitles: false)),
                ),
                borderData: FlBorderData(show: false),
                minX: 0,
                maxX: (chartData.length - 1).toDouble(),
                minY: 0,
                maxY: (maxDuration * 1.2).toDouble(),
                lineBarsData: [
                  LineChartBarData(
                    spots: chartData.asMap().entries.map((entry) {
                      return FlSpot(entry.key.toDouble(), (entry.value['duration'] as int).toDouble());
                    }).toList(),
                    isCurved: true,
                    color: Colors.blue[600],
                    barWidth: 3,
                    isStrokeCapRound: true,
                    belowBarData: BarAreaData(
                      show: true,
                      color: Colors.blue.withOpacity(0.1),
                    ),
                    dotData: FlDotData(
                      show: true,
                      getDotPainter: (spot, percent, barData, index) {
                        return FlDotCirclePainter(
                          radius: 4,
                          color: Colors.white,
                          strokeWidth: 2,
                          strokeColor: Colors.blue[600]!,
                        );
                      },
                    ),
                  ),
                ],
                lineTouchData: LineTouchData(
                  touchTooltipData: LineTouchTooltipData(
                    getTooltipItems: (spots) {
                      return spots.map((spot) {
                        return LineTooltipItem(
                          '${spot.y.toInt()}분',
                          const TextStyle(color: Colors.white, fontWeight: FontWeight.bold, fontSize: 12),
                        );
                      }).toList();
                    },
                  ),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }

  /// 날짜별 그룹화된 로그 카드
  Widget _buildGroupedLogItem(String date, List<Map<String, dynamic>> sessions) {
    final totalDuration = sessions.fold<int>(0, (sum, s) => sum + ((s['duration_min'] as num?)?.toInt() ?? 0));
    final timeRanges = sessions.map((s) => '${s['start_time']} ~ ${s['end_time']}').join(', ');

    return GestureDetector(
      onTap: () => _showSessionManageSheet(date, sessions),
      child: Container(
        margin: const EdgeInsets.only(bottom: 10),
        padding: const EdgeInsets.all(14),
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(12),
          boxShadow: const [BoxShadow(color: Colors.black12, blurRadius: 4)],
        ),
        child: Row(
          children: [
            Container(
              padding: const EdgeInsets.all(10),
              decoration: BoxDecoration(
                color: Colors.green.withOpacity(0.1),
                shape: BoxShape.circle,
              ),
              child: const Icon(Icons.directions_walk, color: Colors.green, size: 24),
            ),
            const SizedBox(width: 14),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    date,
                    style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 15),
                  ),
                  const SizedBox(height: 4),
                  Row(
                    children: [
                      Icon(Icons.access_time, size: 14, color: Colors.grey[500]),
                      const SizedBox(width: 4),
                      Expanded(
                        child: Text(
                          timeRanges,
                          style: TextStyle(color: Colors.grey[600], fontSize: 13),
                          overflow: TextOverflow.ellipsis,
                          maxLines: 2,
                        ),
                      ),
                    ],
                  ),
                ],
              ),
            ),
            const SizedBox(width: 8),
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
              decoration: BoxDecoration(
                color: Colors.blue[50],
                borderRadius: BorderRadius.circular(20),
              ),
              child: Text(
                '${totalDuration}분',
                style: TextStyle(
                  color: Colors.blue[700],
                  fontWeight: FontWeight.bold,
                  fontSize: 13,
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
