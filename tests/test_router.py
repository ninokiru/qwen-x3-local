from app.router import MODE_TO_LORA, route


def test_routes_agent_tasks():
    assert route("Đọc file config.yaml giúp tôi") == "agent"


def test_routes_code_tasks():
    assert route("Viết hàm python tính giai thừa") == "code"


def test_routes_general_chat():
    assert route("Thủ đô của Việt Nam là gì?") == "chat"


def test_substring_does_not_trigger():
    # "profile" không được coi là "file"
    assert route("Cập nhật profile của tôi") == "chat"


def test_every_mode_has_lora_mapping():
    assert set(MODE_TO_LORA) == {"chat", "code", "agent"}


def test_routes_project_modification_to_agent():
    assert route("Sửa code trong file main.py") == "agent"
