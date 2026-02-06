import 'package:flutter/material.dart';
import 'package:fl_chart/fl_chart.dart';

class PetActivityPage extends StatelessWidget {
  const PetActivityPage({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFFF5F7FA),
      body: SingleChildScrollView(
        child: Column(
          children: [
            _buildHeader(context),
            Padding(
              padding: const EdgeInsets.all(16.0),
              child: Column(
                children: [
                  _buildWeightSection(),
                  const SizedBox(height: 16),
                  _buildActivitySection(), // 이제 정의됨
                  const SizedBox(height: 16),
                  _buildTimelineSection(), // 이제 정의됨
                  const SizedBox(height: 24),
                  _buildSaveButton(),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildHeader(BuildContext context) {
    return Container(
      padding: const EdgeInsets.only(top: 50, bottom: 30, left: 20, right: 20),
      decoration: const BoxDecoration(
        color: Color(0xFF3384FF),
        borderRadius: BorderRadius.only(
          bottomLeft: Radius.circular(30),
          bottomRight: Radius.circular(30),
        ),
      ),
      child: Column(
        children: [
          Row(
            children: [
              IconButton(
                icon: const Icon(Icons.arrow_back, color: Colors.white),
                onPressed: () => Navigator.pop(context),
              ),
              const Expanded(
                child: Center(
                  child: Text(
                    "활동량 & 비만도",
                    style: TextStyle(color: Colors.white, fontSize: 22, fontWeight: FontWeight.bold),
                  ),
                ),
              ),
              const SizedBox(width: 48),
            ],
          ),
          const SizedBox(height: 10),
          const Text("2026년 2월 6일 금요일", style: TextStyle(color: Colors.white70)),
        ],
      ),
    );
  }

  Widget _buildWeightSection() {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: _cardDecoration(),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Row(
            children: [
              Icon(Icons.scale, color: Colors.blue),
              SizedBox(width: 8),
              Text("비만도 확인", style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
            ],
          ),
          const SizedBox(height: 20),
          GridView.count(
            shrinkWrap: true,
            physics: const NeverScrollableScrollPhysics(),
            crossAxisCount: 2,
            childAspectRatio: 2.5,
            mainAxisSpacing: 10,
            crossAxisSpacing: 10,
            children: [
              _buildDataTile("현재 체중", "8.2 kg", Colors.black),
              _buildDataTile("체형 지수", "102.5%", Colors.green),
              _buildDataTile("이상 체중", "8 kg", Colors.black),
              _buildDataTile("체고", "52 cm", Colors.black),
            ],
          ),
          const SizedBox(height: 20),
          _buildHealthStatusBox(),
          const SizedBox(height: 20),
          const Text("최근 7일 체중 변화", style: TextStyle(color: Colors.grey)),
          const SizedBox(height: 150, child: WeightLineChart()),
        ],
      ),
    );
  }

  // 콩이의 정상 상태 박스
  Widget _buildHealthStatusBox() {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.symmetric(vertical: 15),
      decoration: BoxDecoration(
        border: Border.all(color: Colors.green),
        borderRadius: BorderRadius.circular(12),
      ),
      child: const Column(
        children: [
          Text("건강 상태", style: TextStyle(color: Colors.green, fontSize: 12)),
          Text("정상", style: TextStyle(color: Colors.green, fontSize: 24, fontWeight: FontWeight.bold)),
        ],
      ),
    );
  }

  // 활동량 섹션 (막대 그래프 들어갈 자리)
  Widget _buildActivitySection() {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: _cardDecoration(),
      child: const Column(
        children: [
          Row(
            children: [
              Icon(Icons.bolt, color: Colors.blue),
              SizedBox(width: 8),
              Text("오늘의 활동량", style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
            ],
          ),
          SizedBox(height: 100, child: Center(child: Text("여기에 BarChart가 들어갑니다"))),
        ],
      ),
    );
  }

  // 타임라인 섹션
  Widget _buildTimelineSection() {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: _cardDecoration(),
      child: const Column(
        children: [
          Row(
            children: [
              Icon(Icons.access_time, color: Colors.blue),
              SizedBox(width: 8),
              Text("하루 활동 타임라인", style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
            ],
          ),
          SizedBox(height: 100, child: Center(child: Text("타임라인 리스트 자리"))),
        ],
      ),
    );
  }

  Widget _buildDataTile(String title, String value, Color valueColor) {
    return Container(
      // 박스 내부 여백을 조절하여 크기감을 줍니다.
      padding: const EdgeInsets.symmetric(vertical: 12, horizontal: 8),
      decoration: BoxDecoration(
        color: const Color(0xFFF8FCF9),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center, // 세로 중앙 정렬
        crossAxisAlignment: CrossAxisAlignment.center, // 가로 중앙 정렬
        children: [
          Text(
            title,
            style: const TextStyle(
              fontSize: 14, // 제목 크기 키움
              color: Colors.grey,
              fontWeight: FontWeight.w500,
            ),
            textAlign: TextAlign.center,
          ),
          const SizedBox(height: 4), // 간격 추가
          Text(
            value,
            style: TextStyle(
              fontSize: 20, // 수치 크기 대폭 키움
              fontWeight: FontWeight.bold,
              color: valueColor,
            ),
            textAlign: TextAlign.center,
          ),
        ],
      ),
    );
  }

  Widget _buildSaveButton() {
    return SizedBox(
      width: double.infinity,
      height: 55,
      child: ElevatedButton(
        onPressed: () {},
        style: ElevatedButton.styleFrom(
          backgroundColor: const Color(0xFF3384FF),
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
        ),
        child: const Text("일기 저장하기", style: TextStyle(fontSize: 18, color: Colors.white)),
      ),
    );
  }

  BoxDecoration _cardDecoration() {
    return BoxDecoration(
      color: Colors.white,
      borderRadius: BorderRadius.circular(20),
      boxShadow: [
        BoxShadow(color: Colors.black.withValues(alpha: 0.05), blurRadius: 10, offset: const Offset(0, 5)),
      ],
    );
  }
}

// 체중 변화 그래프 클래스
class WeightLineChart extends StatelessWidget {
  const WeightLineChart({super.key});

  @override
  Widget build(BuildContext context) {
    return LineChart(
      LineChartData(
        gridData: const FlGridData(show: true, drawVerticalLine: false),
        titlesData: const FlTitlesData(show: false),
        borderData: FlBorderData(show: false),
        lineBarsData: [
          LineChartBarData(
            spots: [
              const FlSpot(0, 8.2),
              const FlSpot(1, 8.3),
              const FlSpot(2, 8.1),
              const FlSpot(3, 8.2),
              const FlSpot(4, 8.0),
              const FlSpot(5, 8.1),
              const FlSpot(6, 8.2),
            ],
            isCurved: false,
            color: Colors.blue,
            barWidth: 3,
            dotData: const FlDotData(show: true),
          ),
        ],
      ),
    );
  }
}