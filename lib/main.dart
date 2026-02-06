import 'package:flutter/material.dart';
import 'package:pet_diary/pet_name_input_page.dart';


void main() => runApp(MaterialApp(home: PetNameInputPage()));

class PetHealthDashboard extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.white,
      appBar: AppBar(
        backgroundColor: Colors.white,
        elevation: 0,
        title: Text('Daily Behavior Diary', style: TextStyle(color: Colors.black, fontSize: 14)),
        centerTitle: true,
        actions: [IconButton(icon: Icon(Icons.share, color: Colors.blue), onPressed: () {})],
      ),
      body: SingleChildScrollView(
        padding: EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // 1. 메인 그라데이션 카드
            _buildHeaderCard(),
            SizedBox(height: 16),

            // 2. 중간 버튼 메뉴
            Row(
              children: [
                _buildActionButton(Icons.book, '일상 일기', '기분 & 활동량', Colors.blue),
                SizedBox(width: 12),
                _buildActionButton(Icons.error_outline, '이상 행동', '건강 체크', Colors.orange),
              ],
            ),
            SizedBox(height: 24),

            // 3. 최근 일기 섹션
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Text('최근 일기', style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
                Text('전체보기 →', style: TextStyle(color: Colors.purple, fontSize: 12)),
              ],
            ),
            SizedBox(height: 12),
            _buildDiaryItem('2026년 2월 6일', '목요일', 85, true),
            _buildDiaryItem('2026년 2월 5일', '수요일', 72, false),
            _buildDiaryItem('2026년 2월 4일', '화요일', 90, false),

            // --- 여기부터 새로 추가된 "건강 트렌드" 섹션입니다 ---
            SizedBox(height: 24),
            Container(
              padding: EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(15),
                boxShadow: [BoxShadow(color: Colors.black12, blurRadius: 4)],
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Icon(Icons.trending_up, color: Colors.green, size: 20),
                      SizedBox(width: 8),
                      Text('이번 주 건강 트렌드', style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16)),
                    ],
                  ),
                  SizedBox(height: 16),
                  _buildTrendRow('평균 활동량', 0.82, Colors.green, '82%'),
                  _buildTrendRow('체중 관리', 0.95, Colors.blue, '95%'),
                  _buildTrendRow('스트레스 관리', 0.88, Colors.purple, '88%'),

                  SizedBox(height: 16),
                  Container(
                    width: double.infinity,
                    padding: EdgeInsets.all(12),
                    decoration: BoxDecoration(
                      color: Colors.green[50],
                      borderRadius: BorderRadius.circular(10),
                      border: Border.all(color: Colors.green[100]!),
                    ),
                    child: Text(
                      '🎉 콩이는 이번 주 매우 건강하게 지냈어요! 활동량과 식사 패턴이 안정적입니다.',
                      style: TextStyle(color: Colors.green[800], fontSize: 13),
                    ),
                  ),
                ],
              ),
            ),
            SizedBox(height: 32),
            Center(
              child: Column(
                children: [
                  Text('AI가 24시간 콩이를 모니터링하고 있어요', style: TextStyle(color: Colors.grey[600], fontSize: 12)),
                  SizedBox(height: 4),
                  Text('8가지 데이터셋 기반 건강 분석 시스템', style: TextStyle(color: Colors.grey[400], fontSize: 11)),
                ],
              ),
            ),
            SizedBox(height: 40),
          ],
        ),
      ),
    );
  }

  // --- 기존 헬퍼 함수들 ---

  Widget _buildHeaderCard() {
    return Container(
      padding: EdgeInsets.all(20),
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(20),
        gradient: LinearGradient(colors: [Colors.purple, Colors.orangeAccent]),
      ),
      child: Column(
        children: [
          Row(
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
          SizedBox(height: 20),
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
        Text(value, style: TextStyle(color: Colors.white, fontSize: 18, fontWeight: FontWeight.bold)),
        Text(label, style: TextStyle(color: Colors.white70, fontSize: 12)),
      ],
    );
  }

  Widget _buildActionButton(IconData icon, String title, String subTitle, Color color) {
    return Expanded(
      child: Container(
        padding: EdgeInsets.symmetric(vertical: 20),
        decoration: BoxDecoration(color: color, borderRadius: BorderRadius.circular(15)),
        child: Column(
          children: [
            Icon(icon, color: Colors.white, size: 30),
            SizedBox(height: 8),
            Text(title, style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold)),
            Text(subTitle, style: TextStyle(color: Colors.white70, fontSize: 10)),
          ],
        ),
      ),
    );
  }

  Widget _buildDiaryItem(String date, String day, int activity, bool hasWarning) {
    return Container(
      margin: EdgeInsets.only(bottom: 12),
      padding: EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(15),
        boxShadow: [BoxShadow(color: Colors.black12, blurRadius: 4)],
      ),
      child: Row(
        children: [
          Container(width: 50, height: 50, decoration: BoxDecoration(color: Colors.grey[300], borderRadius: BorderRadius.circular(8))),
          SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(date, style: TextStyle(fontWeight: FontWeight.bold)),
                Text(day, style: TextStyle(color: Colors.grey, fontSize: 12)),
                Row(
                  children: [
                    Icon(Icons.trending_up, size: 14, color: Colors.green),
                    Text(' 활동 $activity', style: TextStyle(fontSize: 12)),
                  ],
                )
              ],
            ),
          ),
          Icon(Icons.sentiment_satisfied_alt, color: Colors.orange),
        ],
      ),
    );
  }

  // 트렌드 게이지 바를 만드는 함수
  Widget _buildTrendRow(String label, double value, Color color, String percent) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 8.0),
      child: Row(
        children: [
          Expanded(flex: 3, child: Text(label, style: TextStyle(fontSize: 13))),
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
          SizedBox(width: 10),
          Text(percent, style: TextStyle(fontSize: 12, color: color, fontWeight: FontWeight.bold)),
        ],
      ),
    );
  }
}

  // 상단 그라데이션 카드 위젯
  Widget _buildHeaderCard() {
    return Container(
      padding: EdgeInsets.all(20),
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(20),
        gradient: LinearGradient(
          colors: [Colors.purple, Colors.orangeAccent],
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
        ),
      ),
      child: Column(
        children: [
          Row(
            children: [
              CircleAvatar(radius: 25, backgroundColor: Colors.white), // 강아지 이미지 들어갈 곳
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
          SizedBox(height: 20),
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
        Text(value, style: TextStyle(color: Colors.white, fontSize: 18, fontWeight: FontWeight.bold)),
        Text(label, style: TextStyle(color: Colors.white70, fontSize: 12)),
      ],
    );
  }

  // 중앙 버튼 위젯
  Widget _buildActionButton(IconData icon, String title, String subTitle, Color color) {
    return Expanded(
      child: Container(
        padding: EdgeInsets.symmetric(vertical: 20),
        decoration: BoxDecoration(
          color: color,
          borderRadius: BorderRadius.circular(15),
        ),
        child: Column(
          children: [
            Icon(icon, color: Colors.white, size: 30),
            SizedBox(height: 8),
            Text(title, style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold)),
            Text(subTitle, style: TextStyle(color: Colors.white70, fontSize: 10)),
          ],
        ),
      ),
    );
  }

  // 하단 일기 리스트 아이템 위젯
  Widget _buildDiaryItem(String date, String day, int activity, bool hasWarning) {
    return Container(
      margin: EdgeInsets.only(bottom: 12),
      padding: EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(15),
        boxShadow: [BoxShadow(color: Colors.black12, blurRadius: 4, offset: Offset(0, 2))],
      ),
      child: Row(
        children: [
          Container(width: 50, height: 50, decoration: BoxDecoration(color: Colors.grey[300], borderRadius: BorderRadius.circular(8))),
          SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(date, style: TextStyle(fontWeight: FontWeight.bold)),
                Text(day, style: TextStyle(color: Colors.grey, fontSize: 12)),
                Row(
                  children: [
                    Icon(Icons.trending_up, size: 14, color: Colors.green),
                    Text(' 활동 $activity', style: TextStyle(fontSize: 12)),
                    if (hasWarning) ...[
                      SizedBox(width: 8),
                      Container(
                        padding: EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                        decoration: BoxDecoration(color: Colors.orange[100], borderRadius: BorderRadius.circular(4)),
                        child: Text('주의사항', style: TextStyle(color: Colors.orange, fontSize: 10)),
                      )
                    ]
                  ],
                )
              ],
            ),
          ),
          Icon(Icons.sentiment_satisfied_alt, color: Colors.orange),
        ],
      ),
    );
  }

