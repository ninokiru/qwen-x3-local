from app.memory import Memory


def test_search_finds_text_and_ignores_diacritics():
    mem = Memory()
    mem.add("note", "Cổng máy chủ mặc định là 8080")
    mem.add("note", "Thư mục dữ liệu nằm ở data/processed")
    hits = mem.search("cổng máy chủ")
    assert hits and hits[0][1].startswith("Cổng")
    # Truy vấn không dấu vẫn khớp nhờ remove_diacritics
    assert mem.search("cong may chu")
    mem.close()


def test_empty_query_returns_nothing():
    mem = Memory()
    assert mem.search("   ") == []
    mem.close()
