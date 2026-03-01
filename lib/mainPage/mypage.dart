import 'package:flutter/material.dart';

class MyPage extends StatefulWidget {
  const MyPage({super.key});

  @override
  State<MyPage> createState() => _MyPageState();
}

class _MyPageState extends State<MyPage> {
  bool _isPushEnabled = true;
  double _currentWeight = 5.4;

  @override
  Widget build(BuildContext context) {
    return SingleChildScrollView(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // --- 1. 프로필 & 배경 헤더 섹션 ---
          _buildProfileHeader(),

          const SizedBox(height: 10),

          // 나머지 설정 섹션들
          Padding(
            padding: const EdgeInsets.all(16.0),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                _buildSectionTitle('신체 정보 기록'),
                _buildSimpleCard(Icons.monitor_weight_outlined, '현재 몸무게', '$_currentWeight kg', Colors.blue),

                const SizedBox(height: 20),
                _buildSectionTitle('알림 설정 (Detection System)'),
                SwitchListTile(
                  secondary: const Icon(Icons.notifications_active_outlined, color: Colors.orange),
                  title: const Text('이상행동 푸시 알림'),
                  value: _isPushEnabled,
                  activeColor: Colors.orange,
                  onChanged: (val) => setState(() => _isPushEnabled = val),
                ),

                const SizedBox(height: 20),
                _buildSectionTitle('보안 및 계정'),
                _buildListTile(Icons.history, '접근 로그 확인', Colors.teal),
                _buildListTile(Icons.no_accounts_outlined, '계정 관리', Colors.redAccent),

                const SizedBox(height: 20),
                _buildSectionTitle('앱 정보'),
                const ListTile(
                  leading: Icon(Icons.info_outline),
                  title: Text('앱 버전'),
                  trailing: Text('v1.0.1', style: TextStyle(color: Colors.grey)),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  // --- 프로필 헤더 위젯 (배경 + 프로필 사진) ---
  Widget _buildProfileHeader() {
    return Stack(
      alignment: Alignment.center,
      children: [
        Column(
          children: [
            // 배경화면 이미지
            Container(
              height: 180,
              width: double.infinity,
              decoration: BoxDecoration(
                color: Colors.grey[300],
                image: const DecorationImage(
                  // 실제 이미지가 있다면 FileImage나 NetworkImage 사용
                  image: NetworkImage('https://via.placeholder.com/600x300'),
                  fit: BoxFit.cover,
                ),
              ),
              child: Align(
                alignment: Alignment.topRight,
                child: IconButton(
                  padding: const EdgeInsets.only(top: 5, right: 10),
                  icon: const CircleAvatar(
                    backgroundColor: Colors.black26,
                    child: Icon(Icons.camera_alt, color: Colors.white, size: 18),
                  ),
                  onPressed: () {
                    // TODO: 배경 이미지 수정 로직
                  },
                ),
              ),
            ),
            const SizedBox(height: 50), // 프로필 이미지가 걸쳐질 공간
          ],
        ),
        // 프로필 이미지 (CircleAvatar)
        Positioned(
          top: 50, // 배경화면 끝부분에 걸치도록 조정
          child: Stack(
            children: [
              Container(
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  border: Border.all(color: Colors.white, width: 4),
                ),
                child: const CircleAvatar(
                  radius: 50,
                  backgroundColor: Colors.blueGrey,
                  backgroundImage: NetworkImage('data:image/jpeg;base64,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'), // 콩이 사진
                ),
              ),
              // 프로필 수정 버튼
              Positioned(
                bottom: 0,
                right: 0,
                child: GestureDetector(
                  onTap: () {
                    // TODO: 프로필 이미지 수정 로직
                  },
                  child: const CircleAvatar(
                    radius: 16,
                    backgroundColor: Colors.blue,
                    child: Icon(Icons.edit, color: Colors.white, size: 16),
                  ),
                ),
              ),
            ],
          ),
        ),
        // 이름과 이메일은 프로필 이미지 아래로 위치
        Positioned(
          bottom: 0,
          child: Column(
            children: [
              const Text('콩이네', style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
              Text('kong_i_love@email.com', style: TextStyle(color: Colors.grey[600], fontSize: 13)),
            ],
          ),
        ),
      ],
    );
  }

  // 헬퍼 위젯들 (코드 중복 제거)
  Widget _buildSectionTitle(String title) => Padding(
    padding: const EdgeInsets.only(left: 4, bottom: 8),
    child: Text(title, style: const TextStyle(fontSize: 13, fontWeight: FontWeight.bold, color: Colors.blueGrey)),
  );

  Widget _buildSimpleCard(IconData icon, String title, String trailing, Color color) => Card(
    elevation: 0, color: Colors.grey[50],
    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(15)),
    child: ListTile(
      leading: Icon(icon, color: color),
      title: Text(title),
      trailing: Text(trailing, style: const TextStyle(fontWeight: FontWeight.bold)),
    ),
  );

  Widget _buildListTile(IconData icon, String title, Color color) => ListTile(
    leading: Icon(icon, color: color),
    title: Text(title),
    onTap: () {},
  );
}