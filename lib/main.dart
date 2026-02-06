import 'package:flutter/material.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Pet Diary',
      debugShowCheckedModeBanner: false, // 디버그 띠 제거
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
        useMaterial3: true,
      ),
      home: const MyHomePage(title: '반려동물 상태 모니터링'),
    );
  }
}

class MyHomePage extends StatefulWidget {
  const MyHomePage({super.key, required this.title});

  final String title;

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        title: Text(widget.title),
      ),
      body: Column(
        children: [
          // 상단: 실시간 상태 요약 카드 (캡스톤 디자인 용도)
          Container(
            padding: const EdgeInsets.all(16.0),
            width: double.infinity,
            child: Card(
              color: Colors.orange.shade50,
              child: const Padding(
                padding: EdgeInsets.all(20.0),
                child: Column(
                  children: [
                    Text('현재 반려동물 상태', style: TextStyle(fontSize: 18)),
                    SizedBox(height: 10),
                    Text('주의 필요', style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold, color: Colors.red)),
                  ],
                ),
              ),
            ),
          ),
          const Padding(
            padding: EdgeInsets.symmetric(horizontal: 16.0),
            child: Align(alignment: Alignment.centerLeft, child: Text("이상 행동 감지 로그", style: TextStyle(fontWeight: FontWeight.bold))),
          ),
          // 하단: 이상 행동 감지 로그 리스트
          Expanded(
            child: ListView.builder(
              itemCount: 10,
              itemBuilder: (context, index) {
                return ListTile(
                  leading: const Icon(Icons.warning, color: Colors.red),
                  title: const Text('이상 행동 감지: 쓰러짐'),
                  subtitle: const Text('2026-02-06 16:30'),
                  trailing: const Icon(Icons.chevron_right),
                  onTap: () {
                    // 클릭 시 상세 정보 페이지 이동 로직 추가 예정
                  },
                );
              },
            ),
          ),
        ],
      ),
    );
  }
}