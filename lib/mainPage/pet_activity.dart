// import 'package:flutter/material.dart';
// import 'package:fl_chart/fl_chart.dart'; // 그래프 라이브러리
//
// class PetActivityPage extends StatelessWidget {
//   const PetActivityPage({super.key});
//
//   @override
//   Widget build(BuildContext context) {
//     return Scaffold(
//       backgroundColor: const Color(0xFFF5F7FA), // 배경색 (연한 회색/청색)
//       body: SingleChildScrollView(
//         child: Column(
//           children: [
//             _buildHeader(context),
//             Padding(
//               padding: const EdgeInsets.all(16.0),
//               child: Column(
//                 children: [
//                   _buildWeightSection(),
//                   const SizedBox(height: 16),
//                   _buildActivitySection(),
//                   const SizedBox(height: 16),
//                   _buildTimelineSection(),
//                   const SizedBox(height: 24),
//                   _buildSaveButton(),
//                 ],
//               ),
//             ),
//           ],
//         ),
//       ),
//     );
//   }
//
//   // 1. 파란색 상단 헤더
//   Widget _buildHeader(BuildContext context) {
//     return Container(
//       padding: const EdgeInsets.only(top: 50, bottom: 30, left: 20, right: 20),
//       decoration: const BoxDecoration(
//         color: Color(0xFF3384FF), // 메인 파란색
//         borderRadius: BorderRadius.only(
//           bottomLeft: Radius.circular(30),
//           bottomRight: Radius.circular(30),
//         ),
//       ),
//       child: Column(
//         children: [
//           Row(
//             children: [
//               IconButton(
//                 icon: const Icon(Icons.arrow_back, color: White),
//                 onPressed: () => Navigator.pop(context),
//               ),
//               const Expanded(
//                 child: Center(
//                   child: Text(
//                     "활동량 & 비만도",
//                     style: TextStyle(color: Colors.white, fontSize: 22, fontWeight: FontWeight.bold),
//                   ),
//                 ),
//               ),
//               const SizedBox(width: 48), // 대칭을 위한 여백
//             ],
//           ),
//           const SizedBox(height: 10),
//           const Text("2026년 2월 6일 목요일", style: TextStyle(color: Colors.white70)),
//         ],
//       ),
//     );
//   }
//
//   // 2. 비만도 확인 섹션 (4개 그리드 + 그래프)
//   Widget _buildWeightSection() {
//     return Container(
//       padding: const EdgeInsets.all(20),
//       decoration: _cardDecoration(),
//       child: Column(
//         crossAxisAlignment: CrossAxisAlignment.start,
//         children: [
//           const Row(
//             children: [
//               Icon(Icons.scale, color: Colors.blue),
//               SizedBox(width: 8),
//               Text("비만도 확인", style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
//             ],
//           ),
//           const SizedBox(height: 20),
//           // 2x2 그리드 데이터 (간략화)
//           GridView.count(
//             shrinkWrap: true,
//             physics: const NeverScrollableScrollPhysics(),
//             crossAxisCount: 2,
//             childAspectRatio: 2.5,
//             mainAxisSpacing: 10,
//             crossAxisSpacing: 10,
//             children: [
//               _buildDataTile("현재 체중", "8.2 kg", Colors.black),
//               _buildDataTile("체형 지수", "102.5%", Colors.green),
//               _buildDataTile("이상 체중", "8 kg", Colors.black),
//               _buildDataTile("체고", "52 cm", Colors.black),
//             ],
//           ),
//           const SizedBox(height: 20),
//           _buildHealthStatusBox(), // "정상" 박스
//           const SizedBox(height: 20),
//           const Text("최근 7일 체중 변화", style: TextStyle(color: Colors.grey)),
//           const SizedBox(height: 150, child: _WeightLineChart()), // 선 그래프 호출
//         ],
//       ),
//     );
//   }
//
//   // 공통 카드 스타일
//   BoxDecoration _cardDecoration() {
//     return BoxDecoration(
//       color: Colors.white,
//       borderRadius: BorderRadius.circular(20),
//       boxShadow: [
//         BoxShadow(color: Colors.black.withOpacity(0.05), blurRadius: 10, offset: const Offset(0, 5)),
//       ],
//     );
//   }
//
//   // 그리드 내 개별 데이터 타일
//   Widget _buildDataTile(String title, String value, Color valueColor) {
//     return Container(
//       padding: const EdgeInsets.all(10),
//       decoration: BoxDecoration(
//         color: const Color(0xFFF8FCF9),
//         borderRadius: BorderRadius.circular(12),
//       ),
//       child: Column(
//         crossAxisAlignment: CrossAxisAlignment.start,
//         children: [
//           Text(title, style: const TextStyle(fontSize: 12, color: Colors.grey)),
//           Text(value, style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold, color: valueColor)),
//         ],
//       ),
//     );
//   }
//
//   // 3. 하단 일기 저장하기 버튼
//   Widget _buildSaveButton() {
//     return SizedBox(
//       width: double.infinity,
//       height: 55,
//       child: ElevatedButton(
//         onPressed: () {},
//         style: ElevatedButton.styleFrom(
//           backgroundColor: const Color(0xFF3384FF),
//           shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
//         ),
//         child: const Text("일기 저장하기", style: TextStyle(fontSize: 18, color: Colors.white)),
//       ),
//     );
//   }
//
// // 나머지 섹션(_buildActivitySection, _buildHealthStatusBox 등)도 유사한 방식으로 구현...
// }