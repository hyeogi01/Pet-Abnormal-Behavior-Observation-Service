import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'package:flutter_screenutil/flutter_screenutil.dart';

class PhotoGalleryPage extends StatefulWidget {
  final String userId;
  const PhotoGalleryPage({super.key, required this.userId});

  @override
  State<PhotoGalleryPage> createState() => _PhotoGalleryPageState();
}

class _PhotoGalleryPageState extends State<PhotoGalleryPage> {
  final String _baseUrl = 'http://localhost:8080';

  List<Map<String, dynamic>> _albums = [];
  bool _isLoading = true;

  // null이면 앨범 목록, non-null이면 해당 날짜 상세
  String? _selectedDate;

  bool _isSelecting = false;
  final Set<String> _selectedKeys = {};

  static const Map<String, String> _emotionEmoji = {
    'happy': '😊',
    'sad': '😢',
    'angry': '😠',
    'relaxed': '😌',
  };

  String _emoji(String emotion) => _emotionEmoji[emotion.toLowerCase()] ?? '🐾';

  @override
  void initState() {
    super.initState();
    _fetchAlbums();
  }

  Future<void> _fetchAlbums() async {
    setState(() => _isLoading = true);
    try {
      final res = await http.get(Uri.parse('$_baseUrl/api/gallery/albums/${widget.userId}'));
      if (res.statusCode == 200) {
        final decoded = jsonDecode(res.body);
        if (decoded['status'] == 'success') {
          setState(() {
            _albums = List<Map<String, dynamic>>.from(decoded['data'] ?? []);
          });
        }
      }
    } catch (e) {
      debugPrint('Album fetch error: $e');
    } finally {
      setState(() => _isLoading = false);
    }
  }

  // ── 삭제: 앨범 전체 ──────────────────────────────────────────────
  Future<void> _deleteAlbums(List<String> dates) async {
    for (final date in dates) {
      await http.delete(
        Uri.parse('$_baseUrl/api/gallery/${widget.userId}/albums/$date'),
      );
    }
    _exitSelectMode();
    await _fetchAlbums();
  }

  // ── 삭제: 개별 사진 ──────────────────────────────────────────────
  Future<void> _deletePhotos(List<Map<String, dynamic>> items) async {
    await http.delete(
      Uri.parse('$_baseUrl/api/gallery/${widget.userId}/photos'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({'items': items}),
    );
    _exitSelectMode();
    await _fetchAlbums();
  }

  void _exitSelectMode() {
    setState(() {
      _isSelecting = false;
      _selectedKeys.clear();
    });
  }

  void _enterSelectMode(String key) {
    setState(() {
      _isSelecting = true;
      _selectedKeys.add(key);
    });
  }

  void _toggleSelect(String key) {
    setState(() {
      if (_selectedKeys.contains(key)) {
        _selectedKeys.remove(key);
        if (_selectedKeys.isEmpty) _isSelecting = false;
      } else {
        _selectedKeys.add(key);
      }
    });
  }

  // ── 삭제 확인 다이얼로그 ─────────────────────────────────────────
  Future<void> _confirmDeleteAlbums() async {
    final confirmed = await _showConfirmDialog(
      '앨범 ${_selectedKeys.length}개를 삭제할까요?\n사진이 영구적으로 삭제됩니다.',
    );
    if (confirmed) await _deleteAlbums(_selectedKeys.toList());
  }

  Future<void> _confirmDeletePhotos() async {
    final album = _albums.firstWhere((a) => a['date'] == _selectedDate);
    final photos = List<Map<String, dynamic>>.from(album['photos'] ?? []);
    final targets = photos
        .where((p) => _selectedKeys.contains(p['time_key'] as String))
        .map((p) => {
              'date': _selectedDate,
              'time_key': p['time_key'],
              'image_url': p['image_url'],
            })
        .toList();
    final confirmed = await _showConfirmDialog(
      '사진 ${targets.length}장을 삭제할까요?\n영구적으로 삭제됩니다.',
    );
    if (confirmed) {
      await _deletePhotos(targets);
      // 해당 날짜 앨범이 없어진 경우 목록으로 복귀
      final stillExists = _albums.any((a) => a['date'] == _selectedDate);
      if (!stillExists) setState(() => _selectedDate = null);
    }
  }

  Future<bool> _showConfirmDialog(String message) async {
    return await showDialog<bool>(
          context: context,
          builder: (ctx) => AlertDialog(
            content: Text(message),
            actions: [
              TextButton(onPressed: () => Navigator.pop(ctx, false), child: const Text('취소')),
              TextButton(
                onPressed: () => Navigator.pop(ctx, true),
                child: const Text('삭제', style: TextStyle(color: Colors.red)),
              ),
            ],
          ),
        ) ??
        false;
  }

  // ── AppBar ────────────────────────────────────────────────────────
  PreferredSizeWidget _buildAppBar() {
    if (_isSelecting) {
      final isDetail = _selectedDate != null;
      return AppBar(
        backgroundColor: Colors.white,
        elevation: 0,
        leading: IconButton(
          icon: const Icon(Icons.close, color: Colors.black),
          onPressed: _exitSelectMode,
        ),
        title: Text(
          '${_selectedKeys.length}${isDetail ? '장' : '개'} 선택됨',
          style: const TextStyle(color: Colors.black, fontSize: 16, fontWeight: FontWeight.bold),
        ),
        actions: [
          TextButton.icon(
            onPressed: _selectedKeys.isEmpty
                ? null
                : isDetail
                    ? _confirmDeletePhotos
                    : _confirmDeleteAlbums,
            icon: const Icon(Icons.delete_outline, color: Colors.red),
            label: const Text('삭제', style: TextStyle(color: Colors.red)),
          ),
        ],
      );
    }
    return AppBar(
      backgroundColor: Colors.white,
      elevation: 0,
      centerTitle: true,
      leading: _selectedDate != null
          ? IconButton(
              icon: const Icon(Icons.arrow_back_ios, color: Colors.black),
              onPressed: () => setState(() {
                _selectedDate = null;
                _exitSelectMode();
              }),
            )
          : null,
      title: Text(
        _selectedDate ?? '활동 사진첩',
        style: const TextStyle(color: Colors.black, fontWeight: FontWeight.bold, fontSize: 16),
      ),
    );
  }

  // ── Build ─────────────────────────────────────────────────────────
  @override
  Widget build(BuildContext context) {
    return PopScope(
      canPop: !_isSelecting && _selectedDate == null,
      onPopInvokedWithResult: (didPop, _) {
        if (didPop) return;
        if (_isSelecting) {
          _exitSelectMode();
        } else if (_selectedDate != null) {
          setState(() => _selectedDate = null);
        }
      },
      child: Scaffold(
        backgroundColor: Colors.white,
        appBar: _buildAppBar(),
        body: _isLoading
            ? const Center(child: CircularProgressIndicator())
            : _albums.isEmpty
                ? const Center(child: Text('AI가 모아둔 사진 기록이 아직 없습니다.'))
                : _selectedDate == null
                    ? _buildAlbumList()
                    : _buildAlbumDetail(_selectedDate!),
      ),
    );
  }


  // ── 앨범 목록 뷰 ──────────────────────────────────────────────────
  Widget _buildAlbumList() {
    return GridView.builder(
      padding: EdgeInsets.all(8.w),
      gridDelegate: SliverGridDelegateWithFixedCrossAxisCount(
        crossAxisCount: 2,
        crossAxisSpacing: 6.w,
        mainAxisSpacing: 6.h,
        childAspectRatio: 1.0,
      ),
      itemCount: _albums.length,
      itemBuilder: (context, index) {
        final album = _albums[index];
        final date = album['date'] as String;
        final coverUrl = album['cover_image_url'] as String? ?? '';
        final emotion = album['cover_emotion'] as String? ?? 'Unknown';
        final count = album['photo_count'] as int? ?? 0;
        final isSelected = _selectedKeys.contains(date);

        return GestureDetector(
          onTap: _isSelecting
              ? () => _toggleSelect(date)
              : () => setState(() => _selectedDate = date),
          onLongPress: () => _isSelecting ? _toggleSelect(date) : _enterSelectMode(date),
          child: Stack(
            fit: StackFit.expand,
            children: [
              // 커버 이미지
              ClipRRect(
                borderRadius: BorderRadius.circular(8.r),
                child: coverUrl.isNotEmpty
                    ? Image.network(coverUrl, fit: BoxFit.cover,
                        errorBuilder: (_, __, ___) => _placeholder())
                    : _placeholder(),
              ),
              // 선택 오버레이
              if (_isSelecting)
                ClipRRect(
                  borderRadius: BorderRadius.circular(8.r),
                  child: Container(
                    color: isSelected ? Colors.blue.withOpacity(0.35) : Colors.transparent,
                    alignment: Alignment.topRight,
                    padding: EdgeInsets.all(6.w),
                    child: Icon(
                      isSelected ? Icons.check_circle : Icons.radio_button_unchecked,
                      color: Colors.white,
                      size: 22,
                    ),
                  ),
                ),
              // 감정 이모티콘 (좌측상단)
              if (!_isSelecting)
                Positioned(
                  top: 6, left: 6,
                  child: Container(
                    padding: EdgeInsets.symmetric(horizontal: 4.w, vertical: 2.h),
                    decoration: BoxDecoration(
                      color: Colors.black45,
                      borderRadius: BorderRadius.circular(6.r),
                    ),
                    child: Text(_emoji(emotion), style: const TextStyle(fontSize: 14)),
                  ),
                ),
              // 사진 수 배지 (우측상단)
              Positioned(
                top: 6, right: 6,
                child: Container(
                  padding: EdgeInsets.symmetric(horizontal: 5.w, vertical: 2.h),
                  decoration: BoxDecoration(
                    color: Colors.black54,
                    borderRadius: BorderRadius.circular(6.r),
                  ),
                  child: Text('$count장',
                      style: TextStyle(color: Colors.white, fontSize: 10.sp)),
                ),
              ),
              // 날짜 (우측하단)
              Positioned(
                bottom: 0, left: 0, right: 0,
                child: ClipRRect(
                  borderRadius: BorderRadius.only(
                    bottomLeft: Radius.circular(8.r),
                    bottomRight: Radius.circular(8.r),
                  ),
                  child: Container(
                    color: Colors.black45,
                    padding: EdgeInsets.symmetric(horizontal: 6.w, vertical: 4.h),
                    alignment: Alignment.centerRight,
                    child: Text(
                      date,
                      style: TextStyle(color: Colors.white, fontSize: 11.sp, fontWeight: FontWeight.w500),
                    ),
                  ),
                ),
              ),
            ],
          ),
        );
      },
    );
  }

  // ── 앨범 상세 뷰 ──────────────────────────────────────────────────
  Widget _buildAlbumDetail(String date) {
    final album = _albums.firstWhere((a) => a['date'] == date, orElse: () => {});
    final photos = List<Map<String, dynamic>>.from(album['photos'] ?? []);

    if (photos.isEmpty) {
      return const Center(child: Text('사진이 없습니다.'));
    }

    return GridView.builder(
      padding: EdgeInsets.all(4.w),
      gridDelegate: SliverGridDelegateWithFixedCrossAxisCount(
        crossAxisCount: 3,
        crossAxisSpacing: 3.w,
        mainAxisSpacing: 3.h,
      ),
      itemCount: photos.length,
      itemBuilder: (context, index) {
        final photo = photos[index];
        final timeKey = photo['time_key'] as String? ?? '';
        final imageUrl = photo['image_url'] as String? ?? '';
        final emotion = photo['emotion'] as String? ?? 'Unknown';
        final timestamp = photo['timestamp'] as String? ?? '';
        final timeDisplay = timestamp.length >= 16 ? timestamp.substring(11, 16) : timeKey;
        final isSelected = _selectedKeys.contains(timeKey);

        return GestureDetector(
          onTap: _isSelecting
              ? () => _toggleSelect(timeKey)
              : () => _showPhotoViewer(photos, index),
          onLongPress: () => _isSelecting ? _toggleSelect(timeKey) : _enterSelectMode(timeKey),
          child: Stack(
            fit: StackFit.expand,
            children: [
              // 사진
              imageUrl.isNotEmpty
                  ? Image.network(imageUrl, fit: BoxFit.cover,
                      errorBuilder: (_, __, ___) => _placeholder())
                  : _placeholder(),
              // 선택 오버레이
              if (_isSelecting)
                Container(
                  color: isSelected ? Colors.blue.withOpacity(0.35) : Colors.transparent,
                  alignment: Alignment.topRight,
                  padding: EdgeInsets.all(4.w),
                  child: Icon(
                    isSelected ? Icons.check_circle : Icons.radio_button_unchecked,
                    color: Colors.white,
                    size: 18,
                  ),
                ),
              // 감정 이모티콘 (좌측상단)
              if (!_isSelecting)
                Positioned(
                  top: 3, left: 3,
                  child: Text(_emoji(emotion), style: const TextStyle(fontSize: 12)),
                ),
              // 촬영 시간 (우측하단)
              Positioned(
                bottom: 0, left: 0, right: 0,
                child: Container(
                  color: Colors.black38,
                  padding: EdgeInsets.symmetric(horizontal: 3.w, vertical: 2.h),
                  alignment: Alignment.centerRight,
                  child: Text(
                    timeDisplay,
                    style: TextStyle(color: Colors.white, fontSize: 9.sp),
                  ),
                ),
              ),
            ],
          ),
        );
      },
    );
  }

  // ── 전체화면 사진 뷰어 ───────────────────────────────────────────
  void _showPhotoViewer(List<Map<String, dynamic>> photos, int initialIndex) {
    showDialog(
      context: context,
      useSafeArea: false,
      builder: (ctx) => _PhotoViewerDialog(
        photos: photos,
        initialIndex: initialIndex,
        emojiOf: _emoji,
      ),
    );
  }

  Widget _placeholder() => Container(
        color: Colors.grey[300],
        child: const Icon(Icons.image_not_supported, color: Colors.white60, size: 36),
      );
}

// ── 전체화면 뷰어 다이얼로그 (StatefulWidget) ─────────────────────
class _PhotoViewerDialog extends StatefulWidget {
  final List<Map<String, dynamic>> photos;
  final int initialIndex;
  final String Function(String) emojiOf;

  const _PhotoViewerDialog({
    required this.photos,
    required this.initialIndex,
    required this.emojiOf,
  });

  @override
  State<_PhotoViewerDialog> createState() => _PhotoViewerDialogState();
}

class _PhotoViewerDialogState extends State<_PhotoViewerDialog> {
  late int _currentIndex;
  late PageController _pageController;

  @override
  void initState() {
    super.initState();
    _currentIndex = widget.initialIndex;
    _pageController = PageController(initialPage: widget.initialIndex);
  }

  @override
  void dispose() {
    _pageController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final total = widget.photos.length;
    final photo = widget.photos[_currentIndex];
    final emotion = photo['emotion'] as String? ?? 'Unknown';
    final timestamp = photo['timestamp'] as String? ?? '';
    final timeDisplay = timestamp.length >= 16 ? timestamp.substring(11, 16) : '';

    return Dialog(
      insetPadding: EdgeInsets.zero,
      backgroundColor: Colors.black,
      child: SizedBox.expand(
        child: Stack(
          children: [
            // ── 사진 PageView ──────────────────────────────────────
            PageView.builder(
              controller: _pageController,
              itemCount: total,
              onPageChanged: (i) => setState(() => _currentIndex = i),
              itemBuilder: (_, i) {
                final url = widget.photos[i]['image_url'] as String? ?? '';
                return InteractiveViewer(
                  minScale: 1.0,
                  maxScale: 4.0,
                  child: Center(
                    child: url.isNotEmpty
                        ? Image.network(
                            url,
                            fit: BoxFit.contain,
                            errorBuilder: (_, __, ___) => const Icon(
                              Icons.image_not_supported,
                              color: Colors.white38,
                              size: 60,
                            ),
                          )
                        : const Icon(Icons.image_not_supported, color: Colors.white38, size: 60),
                  ),
                );
              },
            ),

            // ── 상단: 닫기 + 페이지 번호 ──────────────────────────
            Positioned(
              top: 0, left: 0, right: 0,
              child: SafeArea(
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    const SizedBox(width: 48),
                    Text(
                      '${ _currentIndex + 1 } / $total',
                      style: const TextStyle(color: Colors.white70, fontSize: 14),
                    ),
                    IconButton(
                      icon: const Icon(Icons.close, color: Colors.white),
                      onPressed: () => Navigator.pop(context),
                    ),
                  ],
                ),
              ),
            ),

            // ── 하단: 감정 이모티콘 + 시간 ────────────────────────
            Positioned(
              bottom: 0, left: 0, right: 0,
              child: Container(
                color: Colors.black54,
                padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
                child: SafeArea(
                  top: false,
                  child: Row(
                    children: [
                      Text(widget.emojiOf(emotion), style: const TextStyle(fontSize: 20)),
                      const SizedBox(width: 10),
                      Text(
                        timeDisplay.isNotEmpty ? timeDisplay : emotion,
                        style: const TextStyle(color: Colors.white70, fontSize: 14),
                      ),
                    ],
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
