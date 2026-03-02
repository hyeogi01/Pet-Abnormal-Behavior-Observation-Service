import 'package:flutter/material.dart';

class MyPage extends StatefulWidget {
  // 1. 정보를 전달받을 변수 추가
  final Map<String, dynamic>? petData;
  const MyPage({super.key, this.petData}); // 생성자에 추가

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
          // 2. 프로필 헤더로 데이터 전달
          _buildProfileHeader(),
          const SizedBox(height: 10),

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

  Widget _buildProfileHeader() {
    // 3. 전달받은 petData에서 이름 추출 (데이터가 없으면 '우리'네로 표시)
    String petName = widget.petData?['pet_name'] ?? '우리';

    return Stack(
      alignment: Alignment.center,
      children: [
        Column(
          children: [
            Container(
              height: 180,
              width: double.infinity,
              decoration: BoxDecoration(
                color: Colors.grey[300],
                image: const DecorationImage(
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
                  onPressed: () {},
                ),
              ),
            ),
            const SizedBox(height: 50),
          ],
        ),
        Positioned(
          top: 50,
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
                  backgroundImage: NetworkImage('https://via.placeholder.com/150'),
                ),
              ),
              Positioned(
                bottom: 0,
                right: 0,
                child: GestureDetector(
                  onTap: () {},
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
        Positioned(
          bottom: 0,
          child: Column(
            children: [
              // 4. 추출한 이름 적용
              Text('${petName}네', style: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
              Text('kong_i_love@email.com', style: TextStyle(color: Colors.grey[600], fontSize: 13)),
            ],
          ),
        ),
      ],
    );
  }

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