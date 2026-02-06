import 'package:flutter/material.dart';

class PageB extends StatelessWidget {
  const PageB({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFFF8F8F8),
      body: Column(
        children: [
          // 1. 상단 주황색 헤더 영역
          _buildHeader(context),

          Expanded(
            child: SingleChildScrollView(
              padding: const EdgeInsets.all(16.0),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  // 2. 오늘 감지된 이상 행동 섹션
                  _buildSectionTitle(Icons.warning_amber_rounded, '오늘 감지된 이상 행동', badgeCount: '2건'),
                  const SizedBox(height: 12),
                  _buildBehaviorCard(
                    time: '14:30',
                    title: '슬개골 관련',
                    description: '걷는 중 약간의 절뚝거림 감지',
                    confidence: 0.78,
                    color: Colors.orange,
                    showImage: true,
                  ),
                  const SizedBox(height: 12),
                  _buildBehaviorCard(
                    time: '10:45',
                    title: '행동 이상',
                    description: '평소보다 활동량 15% 감소',
                    confidence: 0.65,
                    color: Colors.purple,
                    showImage: false,
                  ),

                  const SizedBox(height: 24),

                  // 3. 건강 지표 모니터링 섹션
                  _buildSectionTitle(Icons.favorite_border, '건강 지표 모니터링'),
                  const SizedBox(height: 12),
                  _buildHealthIndicatorBox(),

                  const SizedBox(height: 24),

                  // 4. 슬개골 건강 분석 섹션
                  _buildSectionTitle(Icons.analytics_outlined, '슬개골 건강 분석'),
                  const SizedBox(height: 12),
                  _buildAnalysisCard(),

                  const SizedBox(height: 24),

                  // 5. AI 분석 기술 설명 섹션
                  _buildAITechInfo(),

                  const SizedBox(height: 24),

                  // 6. 일기 저장하기 버튼
                  _buildSaveButton(),
                  const SizedBox(height: 40),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  // --- 위젯 빌더 함수들 ---

  Widget _buildHeader(BuildContext context) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.only(top: 50, left: 16, right: 16, bottom: 20),
      decoration: const BoxDecoration(
        color: Colors.orange,
        borderRadius: BorderRadius.only(bottomLeft: Radius.circular(30), bottomRight: Radius.circular(30)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          GestureDetector(
            onTap: () => Navigator.pop(context),
            child: const Row(
              children: [
                Icon(Icons.arrow_back, color: Colors.white, size: 20),
                SizedBox(width: 4),
                Text('뒤로', style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold)),
              ],
            ),
          ),
          const SizedBox(height: 10),
          const Center(
            child: Text(
              '이상 행동 일기',
              style: TextStyle(color: Colors.white, fontSize: 22, fontWeight: FontWeight.bold),
            ),
          ),
          const SizedBox(height: 8),
          const Center(
            child: Text(
              '2026년 2월 6일 목요일',
              style: TextStyle(color: Colors.white, fontSize: 14),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildSectionTitle(IconData icon, String title, {String? badgeCount}) {
    return Row(
      children: [
        Icon(icon, color: Colors.redAccent, size: 22),
        const SizedBox(width: 8),
        Text(title, style: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
        if (badgeCount != null) ...[
          const Spacer(),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 2),
            decoration: BoxDecoration(color: Colors.orange[100], borderRadius: BorderRadius.circular(10)),
            child: Text(badgeCount, style: const TextStyle(color: Colors.orange, fontSize: 12, fontWeight: FontWeight.bold)),
          ),
        ]
      ],
    );
  }

  Widget _buildBehaviorCard({
    required String time,
    required String title,
    required String description,
    required double confidence,
    required Color color,
    required bool showImage,
  }) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(15),
        border: Border.all(color: color.withOpacity(0.3)),
        boxShadow: const [BoxShadow(color: Colors.black12, blurRadius: 4)],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
                decoration: BoxDecoration(color: color, borderRadius: BorderRadius.circular(4)),
                child: Text(time, style: const TextStyle(color: Colors.white, fontSize: 11, fontWeight: FontWeight.bold)),
              ),
              const SizedBox(width: 8),
              Text(title, style: TextStyle(color: color, fontWeight: FontWeight.bold, fontSize: 16)),
            ],
          ),
          const SizedBox(height: 8),
          Text(description, style: const TextStyle(fontSize: 13, color: Colors.black87)),
          if (showImage) ...[
            const SizedBox(height: 12),
            ClipRRect(
              borderRadius: BorderRadius.circular(10),
              child: Image.network(
                'data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wCEAAkGBwgHBgkIBwgKCgkLDRYPDQwMDRsUFRAWIB0iIiAdHx8kKDQsJCYxJx8fLT0tMTU3Ojo6Iys/RD84QzQ5OjcBCgoKDQwNGg8PGjclHyU3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3N//AABEIAJQAlQMBIgACEQEDEQH/xAAbAAABBQEBAAAAAAAAAAAAAAAEAAECAwUGB//EAC0QAAICAgEEAQQBBAIDAAAAAAABAgMEESEFEjFBUQYTImGBIzJScRSRBxVC/8QAGQEAAgMBAAAAAAAAAAAAAAAAAQIAAwQF/8QAHxEBAQACAwEBAQEBAAAAAAAAAAECEQMSITFBBGFR/9oADAMBAAIRAxEAPwDtZA9j5LpA9r5OZk3xZFk0UwZNMXZhuMzRpZlY75NGmXBq4r4o5IOgy2LBYMuizVGda2Nsi2LYQT2LZHYtkQ42xtiIhNiGYzIhCEMASEIREc1Jg1rL5A1rOZlW+JQfBYmUQfosTENRdDD6ZGZSzW6fjzyHxxFeWaOH1RyeT0RB+C+O/WwmGNVV4Wy1KP8A8pI3TH/rLcg0KpPmXBZGqPsskyHcNopnVH5ZCUNeGTbISfBEQEQc1vljqSftAE42xMYAnGEIiEIYREcxNgtr5LpsFtfJysnQicWWJg8WWxYphmJF22wrT05PR2eNVHHojBfBxeBb9rKqkvUkds5JLjxo3fyyatY/6LfFc5blpvgkuF+iFS3Hu+WPN6NbOacip73wVZebTh1StyLI11rzJ+EB0dVxs3H/AORhZELa5eJwe0S3SDpTSeu9AtuRptdyZm/+xi7nCe2l50Z2b9RdOpyIVXTjXKx6gpTW5fwV5Zw8xrbsv42v5Brc3s9gMrHHU4Pafj4Asm+Tlz4Zny5bF043RY3UVLSb4Zoppra8HFUZDT0348M6XpN/3cfW96H4+TsGeGmgIZaEXqTjDCIjkrJAlsuS2yQFbPk5NdGLoyLoyAo2cl0JiiNrlrwbuB1acIfayNyglxL2jnq5BVUvBZx53C+EzxmX129UlKtOPKaXIpr8TJ6BktuVM3vfK/RtN8b0zqYZTKbc/PHrXHfXX0xf9R4Ea8XLljXQ2ltvtlFrTT17/YJ9EfSV30/0yWHkXKzc3JtPjn4O2nOPvgzOsdSq6fhzyLXquC22GpGb1PDhj0SnTX9ySWu32zw/q3091vP647qIanKf97nr7Z6XZ/5H6RjZElm/chF8KXb3a/ejcoWB1GpZWMuJ8qTXkp5JZdxdhZZqgejRlLChjZL7rK4pd/ju0vINnKVVqh+g+dToluL9+SEMd9QyIwXnXky32a/V08u/xm19yn4Om6FCX23N7UW+CqHRNWxba7fZsU1Qoh9uC4RbxcVl9JyZy/Foww5pUEOMIiOEttAL7kmV35QN2W3y/HemcquhBML1sJruTKKMGS/u5DKsTYuxW1XIMqtT0UQxNB2B023Ks7aYvjzIbCW3ULlZJutLoTnLNrlFNxT5aR0t12pdq4+SGBhLEpUIvnXLLrao2Re1p/J1OPDpjpz+TLtWfZbGTl8L4MX6hux54cqchf05/i9LaTfyat2NOptJPtfIFZRCVclZWnF/PsltSaeC5v00p/UU8Z2wlSpd/wCHnTfs9d6PF4WNCMVpRiuF8F66L06i95FOFRC3/NR5HcXCxa/tKOS2rsIsvsT59PkJ6NQo32W/rQLGiVrSXjZq4tSohw/yfLFwx93RyupoW2Nsr7xd5p2qWbH2V9wu4gJ7ER2MFHnWPheHLlmjTj6WkkH1YevKL446S8HK6Wt/aAoUr2i5Q14CVT+ixVIPQLkqxcSeTYoQ222df0/DrxKlGC51yzL6HS1NtL+TeXCOh/PxzGbY+bO5XR3+iuUuCRXNmhSok3NyT8AOZGPbz/0HTYHk6kiWJHPZtt2t1y7deQWmc29ym2zQyILctrx5Ae3XgyZz1pxvg/EnqSXyH70ZGLL80bFcdpN+yT/EsRTkxOTLNJEWg7oaR7h1YNJDQiHtQ0s7xhdoht0AnYS7P0Edn6JKvgp6LewZV79Eo0LuXnbYT2pF+JR9yanLxEecZLnodh0xpqikuX5CSKH9GqTU0z72aTKpslJlVktLZERm1r+AC6aUmi3IvUYuXcZ7tck5bQuWWqMgfMa2/wDXIBJpIMzZOUU01rXlGdKXozcl9X4LKH/UN+l7qi/0YGMty2btL/px/wBC4GzT0IZsWywppeBRQpPwSDpCYhmxBKu1ob2S02SUV/IdBtBR2aFEdRQPVFOXPgKjxwWYwmSbYt8FcpChPcd7GKayWkAZFsk+Au6XBnZXOgZfEjLz7JzXbKX4/BCqfbBfCWmSyauWnLwZGZnOm/7UVva8mXLLV9aMcdwdk3JRbXBTGcbPKM5zndNNvhegypNaKu26s66GUaUl8GrGz8UZNT1ywiF2w9tJoepbJqQGreCcZjzIuhW1slsGU+SfcPKWrHIRV3CDsNNJLQ+htjeSxWnCWnwSsvlGHCfBWlolvu49B2lB25703JOOvI+D1CFt6pXlhE6YSWnFMqhjV12qyMUpfKBvLaeaGWLgByIvtNB8oFvXA9hY5/OUuds57LjL/kQm+V4Op6lFJHM5b1NL9mHl8rXx/F1Veu3fsJi+SuOpRj/otnHWmhNG/Tzs1HRGuxryV3NprjhkFz4EtPIOjb+wiu0zYvRdGTDjkFjRVhYp/sAjN/JNWP5LZkrsG944MpiH7F030SQhGhQZ+SS8CEFKf0RYhEBdHmJTeuBCHnwv6x+oLaOU6m+xx1/l7EIxc31q4vg7G5jDYbZXF649CECfDX6ByH+aj6GX4+BhGerV0IrW/ZNJCEGJVkUTiIQ6tMQhBK//2Q==', // 샘플 이미지
                height: 600,
                width: double.infinity,
                fit: BoxFit.cover,
              ),
            ),
          ],
          const SizedBox(height: 12),
          Row(
            children: [
              const Text('AI 신뢰도', style: TextStyle(fontSize: 12, color: Colors.grey)),
              const SizedBox(width: 8),
              Expanded(
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(10),
                  child: LinearProgressIndicator(
                    value: confidence,
                    backgroundColor: Colors.grey[200],
                    valueColor: AlwaysStoppedAnimation<Color>(color),
                    minHeight: 8,
                  ),
                ),
              ),
              const SizedBox(width: 10),
              Text('${(confidence * 100).toInt()}%', style: TextStyle(fontSize: 12, color: color, fontWeight: FontWeight.bold)),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildHealthIndicatorBox() {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(15),
        boxShadow: const [BoxShadow(color: Colors.black12, blurRadius: 4)],
      ),
      child: Column(
        children: [
          _buildIndicatorRow('슬개골 건강도', 75, Colors.orange, '보통의 주의가 필요합니다'),
          const Divider(height: 24),
          _buildIndicatorRow('스트레스 지수', 35, Colors.green, '매우 양호합니다'),
          const Divider(height: 24),
          _buildIndicatorRow('안구 건강', 92, Colors.green, '건강한 상태입니다'),
          const Divider(height: 24),
          _buildIndicatorRow('피부 상태', 88, Colors.green, '양호한 상태입니다'),
        ],
      ),
    );
  }

  Widget _buildIndicatorRow(String label, int value, Color color, String subText) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Text(label, style: const TextStyle(fontWeight: FontWeight.bold)),
            Text('$value', style: TextStyle(color: color, fontSize: 18, fontWeight: FontWeight.bold)),
          ],
        ),
        const SizedBox(height: 8),
        ClipRRect(
          borderRadius: BorderRadius.circular(10),
          child: LinearProgressIndicator(
            value: value / 100,
            backgroundColor: Colors.grey[100],
            valueColor: AlwaysStoppedAnimation<Color>(color),
            minHeight: 10,
          ),
        ),
        const SizedBox(height: 4),
        Text(subText, style: const TextStyle(fontSize: 11, color: Colors.grey)),
      ],
    );
  }

  Widget _buildAnalysisCard() {
    return Container(
      width: double.infinity,
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(15),
        boxShadow: const [BoxShadow(color: Colors.black12, blurRadius: 4)],
      ),
      child: Column(
        children: [
          Container(
            padding: const EdgeInsets.all(16),
            decoration: BoxDecoration(color: Colors.orange[50], borderRadius: const BorderRadius.vertical(top: Radius.circular(15))),
            child: Row(
              children: [
                const Icon(Icons.camera_alt, color: Colors.orange, size: 24),
                const SizedBox(width: 10),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Text('보행 영상 분석 결과', style: TextStyle(fontWeight: FontWeight.bold, fontSize: 15)),
                      const SizedBox(height: 4),
                      const Text('오늘 오후 2시 30분, 콩이의 걸음걸이에서 약간의 불규칙성이 감지되었습니다.', style: TextStyle(fontSize: 12, height: 1.5)),
                      const SizedBox(height: 8),
                      Row(
                        children: [
                          _buildLegStatus('왼쪽 뒷다리', '주의', Colors.orange),
                          const SizedBox(width: 10),
                          _buildLegStatus('오른쪽 뒷다리', '정상', Colors.green),
                        ],
                      )
                    ],
                  ),
                )
              ],
            ),
          ),
          Container(
            padding: const EdgeInsets.all(16),
            child: const Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    CircleAvatar(radius: 12, backgroundColor: Colors.blueAccent, child: Icon(Icons.person, color: Colors.white, size: 14)),
                    SizedBox(width: 8),
                    Text('수의사 권장사항', style: TextStyle(fontWeight: FontWeight.bold, fontSize: 14)),
                  ],
                ),
                SizedBox(height: 8),
                Text('• 무리한 점프나 계단 오르내리기를 줄여주세요', style: TextStyle(fontSize: 12, height: 1.6)),
                Text('• 체중 관리를 통해 관절 부담을 줄여주세요', style: TextStyle(fontSize: 12, height: 1.6)),
                Text('• 증상이 지속되면 동물병원 방문을 권장합니다', style: TextStyle(fontSize: 12, height: 1.6)),
              ],
            ),
          )
        ],
      ),
    );
  }

  Widget _buildLegStatus(String leg, String status, Color color) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
      decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(4), border: Border.all(color: color.withOpacity(0.5))),
      child: Row(
        children: [
          Text('$leg: ', style: const TextStyle(fontSize: 11)),
          Text(status, style: TextStyle(fontSize: 11, color: color, fontWeight: FontWeight.bold)),
        ],
      ),
    );
  }

  Widget _buildAITechInfo() {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        gradient: const LinearGradient(colors: [Color(0xFF8E2DE2), Color(0xFFF64C75)]),
        borderRadius: BorderRadius.circular(15),
      ),
      child: const Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(Icons.description_outlined, color: Colors.white, size: 20),
              SizedBox(width: 8),
              Text('AI 분석 기술', style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold)),
            ],
          ),
          SizedBox(height: 8),
          Text(
            '이 일기는 8가지 데이터셋을 활용한 AI 분석으로 작성됩니다. 반려동물 보행영상, 행동 데이터, 건강정보, 관절 인식, 안구/피부 증상 이미지, X-ray 분석 등을 통해 콩이의 건강을 24시간 모니터링합니다.',
            style: TextStyle(color: Colors.white, fontSize: 11, height: 1.5),
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
          backgroundColor: Colors.orange,
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
        ),
        child: const Text('일기 저장하기', style: TextStyle(color: Colors.white, fontSize: 18, fontWeight: FontWeight.bold)),
      ),
    );
  }
}