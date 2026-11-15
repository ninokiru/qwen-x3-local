"""Sinh thêm mẫu train dạng JSONL từ template (chat, code, tool_call).

Mọi đáp án code/tool được tính bằng Python thật nên không có mẫu "sai lệch".
Seed cố định nên chạy lại cho ra đúng cùng một bộ dữ liệu.

    python -m data.generate_dataset --out-dir data/generated

Ba file: chat_vi.jsonl, code_vi.jsonl, tool_call.jsonl. Dùng
`data/prepare_dataset.py --inputs data/raw/*.jsonl data/generated/*.jsonl` để trộn.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

Record = dict
Maker = Callable[[random.Random], Record]


@dataclass
class Template:
    """Một khuôn sinh mẫu. count là số mẫu cần sinh từ khuôn này."""
    name: str
    count: int
    make: Maker
    seen: set = field(default_factory=set)


def _user(text: str) -> dict:
    return {"role": "user", "content": text}


def _assistant(text: str) -> dict:
    return {"role": "assistant", "content": text}


def _system_tool() -> dict:
    return {
        "role": "system",
        "content": (
            "Bạn là Qwen-X3, trợ lý AI thông minh. Bạn có thể gọi công cụ để thực hiện tác vụ.\n\n"
            "Công cụ có sẵn:\n"
            "- read_file(path): đọc nội dung file\n"
            "- write_file(path, content): ghi nội dung vào file\n"
            "- list_dir(path): liệt kê thư mục\n"
            "- run_python(code): chạy code Python\n\n"
            "Để gọi công cụ, dùng định dạng:\n<tool_call>\n"
            '{"name": "tên_công_cụ", "arguments": {"tham_số": "giá_trị"}}\n</tool_call>'
        ),
    }


def tool_call_block(name: str, arguments: dict) -> str:
    return "<tool_call>\n" + json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False) + "\n</tool_call>"


def _pick(rng: random.Random, pool: list) -> object:
    return rng.choice(pool)


def _num(rng: random.Random, lo: int = 1, hi: int = 100) -> int:
    return rng.randint(lo, hi)


def _fmt_list(xs) -> str:
    return "[" + ", ".join(str(x) for x in xs) + "]"


def _pad(n: int, width: int = 2) -> str:
    return str(n).zfill(width)


# --------------------------------------------------------------------------- chat

PLACES = [
    ("Hà Nội", "thủ đô của Việt Nam, trung tâm chính trị và văn hoá", "phố cổ Hồ Gươm", "ra Hồ Gươm, phố cổ và thưởng thức cà phê trứng"),
    ("Huế", "cố đô của Việt Nam, nổi tiếng với Đại Nội và sông Hương", "Đại Nội", "thăm Đại Nội, chùa Thiên Mụ và ăn bún bò Huế"),
    ("Đà Nẵng", "thành phố lớn miền Trung, có cầu Rồng và bãi biển Mỹ Khê", "cầu Rồng", "tắm biển Mỹ Khê, đi Bà Nà và ăn mì Quảng"),
    ("Cần Thơ", "trung tâm đồng bằng sông Cửu Long, có chợ nổi Cái Răng", "chợ nổi Cái Răng", "đi chợ nổi sáng sớm và thăm bến Ninh Kiều"),
    ("Hải Phòng", "thành phố cảng lớn nhất miền Bắc, gắn với hoa phượng đỏ", "đồ Sơn", "tắm biển Đồ Sơn và ăn bánh mì cay"),
    ("Đà Lạt", "thành phố cao nguyên ở Lâm Đồng, khí hậu mát quanh năm", "hồ Xuân Hương", "dạo hồ Xuân Hương và thăm vườn hoa"),
    ("Nha Trang", "thành phố biển ở Khánh Hoà, nổi tiếng với vịnh và hải sản", "Vinpearl", "lặn biển ở Hòn Mun và ăn hải sản"),
    ("Quy Nhơn", "thành phố biển ở Bình Định, có Eo Gió và Kỳ Co", "Eo Gió", "đi Eo Gió, Kỳ Co và ăn bánh xèo tôm nhảy"),
    ("Sa Pa", "thị xã vùng cao ở Lào Cai, nổi tiếng với ruộng bậc thang", "Fansipan", "trek ruộng bậc thang và lên đỉnh Fansipan"),
    ("Hạ Long", "thành phố ở Quảng Ninh, gắn với vịnh di sản thiên nhiên thế giới", "vịnh Hạ Long", "đi thuyền ngắm hang Sửng Sốt và ăn chả mực"),
]

DISHES = [
    ("phở bò", "nước dùng xương bò hầm với quế hồi, ăn cùng bánh phở và thịt bò"),
    ("bún bò Huế", "nước dùng cay, có mắm ruốc và thịt bò hoặc chân giò"),
    ("bánh mì", "vỏ giòn, nhân thịt, pate, rau thơm và đồ chua"),
    ("cơm tấm", "cơm gạo tấm ăn với sườn nướng, bì và chả trứng"),
    ("bánh xèo", "bánh bột gạo chiên giòn, ăn cuốn với rau sống và nước chấm"),
    ("gỏi cuốn", "cuốn bánh tráng với tôm thịt, rau thơm, chấm nước mắm chua ngọt"),
    ("chả cá", "cá xay nhuyễn ướp nghệ và thì là, rán hoặc nấu canh"),
    ("bún chả", "bún ăn với chả nướng, nước chấm chua ngọt và rau sống"),
]

TOPICS = [
    ("Python", "ngôn ngữ lập trình bậc cao, dễ đọc, dùng rộng cho web, dữ liệu và tự động hoá"),
    ("SQL", "ngôn ngữ truy vấn cơ sở dữ liệu quan hệ, dùng để tạo bảng và lấy dữ liệu"),
    ("Git", "hệ quản lý phiên bản phân tán, cho phép tạo nhánh và gộp thay đổi"),
    ("Docker", "nền tảng đóng gói ứng dụng kèm môi trường chạy để triển khai nhất quán"),
    ("HTTP", "giao thức trao đổi dữ liệu giữa trình duyệt và máy chủ, dạng request và response"),
    ("JSON", "định dạng dữ liệu dạng văn bản, dùng cặp key và value để trao đổi giữa các hệ thống"),
    ("REST API", "kiểu thiết kế API dựa trên tài nguyên và các phương thức HTTP"),
    ("JWT", "token chứa thông tin người dùng đã ký số, dùng để xác thực không trạng thái"),
]

HABITS = [
    ("uống đủ nước", "giúp cơ thể điều hoà nhiệt độ, hỗ trợ tiêu hoá và giảm mệt mỏi"),
    ("ngủ đủ giấc", "giúp não củng cố ký ức, phục hồi cơ thể và giữ tinh thần ổn định"),
    ("tập thể dục đều", "giúp tim mạch tốt hơn, tăng sức bền và giảm căng thẳng"),
    ("đọc sách mỗi ngày", "mở rộng vốn từ, tăng khả năng tập trung và mở rộng hiểu biết"),
    ("ngồi đúng tư thế", "giảm áp lực lên cổ, vai và lưng dưới"),
    ("nghỉ mắt 20 phút một lần", "giảm khô mắt và đau đầu khi làm việc với màn hình"),
]

SECTIONS = ["workspace", ".", "src", "docs", "scripts", "tests", "configs", "data", "models", "logs"]
FILES = ["config.json", "app.py", "utils.py", "main.py", "readme.txt", "notes.md",
         "data.csv", "requirements.txt", "settings.yaml", "report.json", "todo.txt", "bai_tap.py"]


def make_place_fact(rng: random.Random) -> Record:
    name, desc, spot, activity = _pick(rng, PLACES)
    kind = rng.randrange(4)
    if kind == 0:
        q = _pick(rng, [f"{name} là gì?", f"Cho tôi biết vài nét về {name}", f"Giới thiệu ngắn gọn về {name}",
                        f"{name} có những gì đáng chú ý?", f"Tóm tắt về {name} cho người chưa đến"])
        a = f"{name} là {desc}."
    elif kind == 1:
        q = _pick(rng, [f"Đi {name} thì nên ghé đâu?", f"Ở {name} có điểm gì nổi bật?", f"Chơi gì ở {name}?",
                        f"Địa điểm nổi tiếng của {name} là gì?", f"Nên ghé thăm gì khi tới {name}?"])
        a = f"Ở {name} bạn nên {activity}. Điểm nhấn thường được nhắc là {spot}."
    elif kind == 2:
        q = _pick(rng, [f"Đến {name} nên làm gì?", f"{name} có gì để khám phá?", f"Lịch trình ngắn ở {name}?",
                        f"Một ngày ở {name} thì làm gì?", f"Kinh nghiệm đi {name}"])
        a = f"{name} là {desc}. Một buổi đủ để {activity}."
    else:
        q = _pick(rng, [f"{name} ở đâu?", f"Nếu đi du lịch {name} thì bắt đầu từ đâu?",
                        f"Tại sao khách du lịch đến {name}?", f"Điểm đáng nhớ nhất của {name}?"])
        a = f"{name} có đặc trưng là {spot}; du khách thường {activity}. {name} là {desc}."
    return {"messages": [_user(q), _assistant(a)]}


def make_dish(rng: random.Random) -> Record:
    dish, desc = _pick(rng, DISHES)
    eat = _pick(rng, ["bữa sáng", "bữa trưa", "bữa tối", "bữa ăn vặt chiều"])
    where = _pick(rng, ["quán ven đường", "quán trong ngõ nhỏ", "hàng lâu năm ở trung tâm", "quán tỉnh lẻ"])
    kind = rng.randrange(4)
    if kind == 0:
        q = _pick(rng, [f"{dish} là món gì?", f"Mô tả {dish} cho tôi nghe", f"{dish} gồm những gì đặc trưng?",
                        f"Kể về {dish}", f"{dish.capitalize()} được nấu thế nào?"])
        a = f"{dish.capitalize()} là món có {desc}. Đây là món quen thuộc ở Việt Nam và mỗi vùng có biến thể riêng."
    elif kind == 1:
        q = _pick(rng, [f"{dish.capitalize()} ăn vào lúc nào là hợp nhất?", f"Nên ăn {dish} lúc nào?",
                        f"{dish} dùng cho bữa nào hợp?"])
        a = f"{dish.capitalize()} hợp nhất cho {eat}. Món này có {desc}; nên chọn {where} vì hương vị thường đậm hơn."
    elif kind == 2:
        q = _pick(rng, [f"Ăn {dish} ở đâu ngon?", f"Tìm quán {dish} đúng vị thì để ý gì?",
                        f"Chọn quán {dish} thế nào?"])
        a = f"Để ăn {dish} đúng vị, hãy chọn {where} và gọi vào {eat}. Món có {desc}, nên ăn ngay khi còn nóng."
    else:
        q = _pick(rng, [f"Ai đến Việt Nam có nên thử {dish}?", f"{dish} có cay không?",
                        f"Vì sao {dish} được yêu thích?"])
        a = f"Nên thử: {dish} có {desc}, giá bình dân và dễ tìm ở {where}. Hương vị khác nhau theo từng vùng và thường dùng vào {eat}."
    return {"messages": [_user(q), _assistant(a)]}


def make_concept(rng: random.Random) -> Record:
    name, desc = _pick(rng, TOPICS)
    angle = _pick(rng, [
        f"{name} là {desc}.",
        f"{name} là {desc}. Khái niệm này xuất hiện nhiều khi làm việc nhóm vì giúp thống nhất cách trao đổi.",
        f"{name} là {desc}. Khi mới học, nên bắt đầu từ ví dụ nhỏ nhất có thể chạy được.",
        f"{name} là {desc}. Cần phân biệt khái niệm này với các khái niệm liền kề để tránh nhầm khi trao đổi.",
    ])
    q = _pick(rng, [
        f"{name} là gì?",
        f"Giải thích {name} cho người mới bắt đầu",
        f"{name} dùng để làm gì?",
        f"Khi nào tôi cần đến {name}?",
        f"Nói ngắn gọn về {name}",
        f"{name} có khó học không?",
        f"Mô tả {name} trong một câu",
        f"Tôi đang tìm hiểu {name}, nên đọc gì trước?",
    ])
    return {"messages": [_user(q), _assistant(angle)]}


def make_habit(rng: random.Random) -> Record:
    habit, benefit = _pick(rng, HABITS)
    extra = _pick(rng, [
        "Nên duy trì đều đặn thay vì làm dồn dập.",
        "Bắt đầu từ mức dễ rồi tăng dần để tránh bỏ giữa chừng.",
        "Ghi lại việc đã làm mỗi tuần để thấy tiến bộ.",
        "Nếu gián đoạn vài ngày thì cứ tiếp tục lại, đừng coi là thất bại.",
        "Ai có bệnh nền nên hỏi bác sĩ trước khi duy trì lâu dài.",
    ])
    q = _pick(rng, [
        f"Vì sao nên {habit}?",
        f"{habit.capitalize()} có lợi ích gì?",
        f"Lợi ích của việc {habit} là gì?",
        f"Tôi có nên {habit} không?",
        f"Làm sao để duy trì thói quen {habit}?",
        f"{habit.capitalize()} giúp gì cho cơ thể?",
        f"Có cần {habit} không?",
        f"{habit.capitalize()} có hại gì không?",
        f"Nhắc tôi lý do để {habit}",
    ])
    return {"messages": [_user(q), _assistant(f"Việc {habit} {benefit}. {extra}")]}


def make_arithmetic(rng: random.Random) -> Record:
    a, b, c = _num(rng, 10, 999), _num(rng, 10, 999), _num(rng, 2, 20)
    kind = rng.choice(["sum", "mult", "percent", "avg", "pow", "sub", "div"])
    if kind == "sum":
        expr, ans = f"{a} + {b}", a + b
    elif kind == "mult":
        expr, ans = f"{a} × {b}", a * b
    elif kind == "sub":
        expr, ans = f"{max(a, b)} - {min(a, b)}", abs(a - b)
    elif kind == "div":
        expr, ans = f"{a * b} ÷ {b}", a
    elif kind == "percent":
        expr, ans = f"{c}% của {a}", a * c / 100
    elif kind == "avg":
        expr, ans = f"trung bình cộng của {a}, {b} và {a + b}", (a + b + a + b) / 3
    else:
        expr, ans = f"bình phương của {a}", a ** 2
    q = _pick(rng, [f"Tính {expr}", f"{expr.capitalize()} bằng bao nhiêu?", f"Bao nhiêu là {expr}?",
                    f"Cho tôi biết {expr} là bao nhiêu"])
    ans_text = f"{ans:g}" if isinstance(ans, float) else str(ans)
    return {"messages": [_user(q), _assistant(f"Kết quả của {expr} là **{ans_text}**.")]}


def make_math_explain(rng: random.Random) -> Record:
    import math
    a, b = _num(rng, 2, 200), _num(rng, 2, 200)
    kind = rng.randrange(4)
    if kind == 0:
        is_prime = a > 1 and all(a % i for i in range(2, int(a ** 0.5) + 1))
        d = next((i for i in range(2, a) if a % i == 0), a)
        q = _pick(rng, [f"{a} là số nguyên tố hay hợp số?", f"{a} có phải số nguyên tố không?",
                        f"Kiểm tra xem {a} là nguyên tố hay hợp số"])
        ans = f"{a} là số nguyên tố." if is_prime else f"{a} là hợp số vì chia hết cho {d}."
    elif kind == 1:
        d = math.gcd(a, b)
        q = _pick(rng, [f"Ước chung lớn nhất của {a} và {b} là bao nhiêu?", f"Tìm UCLN({a}, {b})",
                        f"UCLN của {a} và {b}?"])
        ans = f"Ước chung lớn nhất của {a} và {b} là {d}."
    elif kind == 2:
        d = math.lcm(a, b)
        q = _pick(rng, [f"Bội chung nhỏ nhất của {a} và {b} là bao nhiêu?", f"Tìm BCNN({a}, {b})",
                        f"BCNN của {a} và {b}?"])
        ans = f"Bội chung nhỏ nhất của {a} và {b} là {d}."
    else:
        d = a * b
        q = _pick(rng, [f"{a} nhân {b} bằng bao nhiêu?", f"Tích của {a} và {b} là gì?",
                        f"{a} × {b} = ?"])
        ans = f"Tích của {a} và {b} là {d}."
    return {"messages": [_user(q), _assistant(ans)]}


def make_conversion(rng: random.Random) -> Record:
    kind = rng.choice(["km", "kg", "c_to_f", "hour", "m_to_cm", "g_to_mg"])
    n = _num(rng, 1, 500)
    if kind == "km":
        src, dst, ans, factor = f"{n} km", "mét (m)", n * 1000, 1000
    elif kind == "kg":
        src, dst, ans, factor = f"{n} kg", "gam (g)", n * 1000, 1000
    elif kind == "c_to_f":
        src, dst, ans, factor = f"{n} độ C", "độ F", round(n * 9 / 5 + 32, 1), None
    elif kind == "hour":
        src, dst, ans, factor = f"{n} giờ", "phút", n * 60, 60
    elif kind == "m_to_cm":
        src, dst, ans, factor = f"{n} m", "xăng-ti-mét (cm)", n * 100, 100
    else:
        src, dst, ans, factor = f"{n} g", "miligam (mg)", n * 1000, 1000
    q = _pick(rng, [f"{src} bằng bao nhiêu {dst}?", f"Đổi {src} sang {dst}",
                    f"{src} là bao nhiêu {dst}?", f"Quy đổi {src} ra {dst} giúp tôi"])
    body = f"{(ans if isinstance(ans, float) else ans):g} {dst}"
    if factor:
        body += f" (nhân với {factor})"
    return {"messages": [_user(q), _assistant(f"{src} tương đương **{body}**.")]}


def make_long_form(rng: random.Random) -> Record:
    order = rng.sample(range(5), 5)
    topic, steps_all = _pick(rng, [
        ("cách học một ngôn ngữ lập trình mới",
         ["chọn một ngôn ngữ phù hợp mục tiêu", "học cú pháp cơ bản qua ví dụ nhỏ",
          "làm 3-5 bài tập nhỏ mỗi ngày", "làm một dự án nhỏ hoàn chỉnh", "đọc lại code cũ và sửa lỗi"]),
        ("cách chuẩn bị cho một buổi phỏng vấn kỹ thuật",
         ["ôn lại cấu trúc dữ liệu cơ bản", "luyện giải bài trên giấy",
          "chuẩn bị câu chuyện về dự án đã làm", "tìm hiểu sản phẩm của công ty", "ngủ đủ và đến sớm"]),
        ("cách giữ cho máy tính chạy nhanh",
         ["gỡ phần mềm không dùng", "dọn file tạm định kỳ",
          "hạn chế chương trình tự khởi động", "cập nhật hệ điều hành và trình duyệt", "thêm RAM nếu vẫn chậm"]),
        ("cách tiết kiệm tiền hàng tháng",
         ["ghi lại mọi khoản chi", "đặt hạn mức cho từng nhóm chi",
          "trả nợ lãi cao trước", "trích một phần thu nhập để tiết kiệm ngay đầu tháng", "xem lại bảng chi mỗi tuần"]),
        ("cách bắt đầu viết nhật ký",
         ["chọn khung giờ cố định mỗi ngày", "bắt đầu bằng 3 câu ngắn",
          "không cần viết đẹp, chỉ cần đều", "ghi cả việc nhỏ và cảm xúc", "đọc lại mỗi tuần để thấy tiến bộ"]),
    ])
    keep = rng.choice([3, 4, 5])
    steps = [steps_all[i] for i in order[:keep]]
    lines = "\n".join(f"{i}. {s}" for i, s in enumerate(steps, 1))
    q = _pick(rng, [f"Cho tôi vài gợi ý về {topic}?", f"Bắt đầu {topic} từ đâu?",
                    f"Những bước cơ bản trong {topic} là gì?",
                    f"Có kế hoạch nào cho {topic} không?",
                    f"Người mới nên làm gì với {topic}?",
                    f"Chia sẻ kinh nghiệm về {topic} giúp tôi"])
    closing = _pick(rng, [
        "Quan trọng là làm đều đặn và điều chỉnh sau mỗi giai đoạn, đừng cố làm hết trong một lần.",
        "Chọn một việc làm được ngay hôm nay rồi mới thêm việc tiếp theo.",
        "Sau 2 tuần hãy xem lại và bỏ bước không mang lại kết quả.",
    ])
    return {"messages": [_user(q), _assistant(f"Có mấy bước thực tế:\n\n{lines}\n\n{closing}")]}


def make_compare(rng: random.Random) -> Record:
    pair, a, b = _pick(rng, [
        ("list và tuple", "list thay đổi được, tuple không thay đổi sau khi tạo",
         "tuple nhanh và nhẹ hơn list một chút, dùng cho dữ liệu cố định"),
        ("HTTP và HTTPS", "HTTPS là HTTP có mã hoá TLS", "HTTPS bảo vệ dữ liệu trên đường truyền, HTTP thì không"),
        ("TCP và UDP", "TCP có bắt tay và đảm bảo thứ tự, UDP thì không",
         "TCP phù hợp khi cần đủ dữ liệu, UDP phù hợp khi cần nhanh như video"),
        ("sách giấy và sách điện tử", "sách giấy dễ tập trung, sách điện tử tiện tìm kiếm và mang theo",
         "nên chọn theo nhu cầu đọc sâu hay tra cứu nhanh"),
        ("GET và POST", "GET lấy dữ liệu và gắn tham số vào URL, POST gửi dữ liệu trong phần thân request",
         "GET phù hợp tra cứu, POST phù hợp tạo mới hoặc gửi dữ liệu nhạy cảm"),
        ("SQL và NoSQL", "SQL dùng bảng có quan hệ chặt, NoSQL linh hoạt hơn về lược đồ",
         "SQL hợp với dữ liệu giao dịch, NoSQL hợp với dữ liệu phi cấu trúc hoặc dung lượng lớn"),
        ("xâu chuỗi và đệ quy", "xâu chuỗi (recursion) gọi lại chính hàm, vòng lặp lặp bằng biến đếm",
         "đệ quy dễ đọc cho bài toán chia nhỏ, vòng lặp an toàn hơn với dữ liệu lớn"),
        ("virtualenv và Docker", "virtualenv chỉ tách môi trường Python, Docker đóng gói cả hệ điều hành thu nhỏ",
         "virtualenv nhẹ cho phát triển, Docker nhất quán hơn khi triển khai"),
    ])
    q = _pick(rng, [
        f"Sự khác nhau giữa {pair} là gì?",
        f"So sánh {pair} giúp tôi với",
        f"{pair.capitalize()}: nên chọn cái nào?",
        f"Khi nào dùng {pair.split(' và ')[0]} thay vì {pair.split(' và ')[1]}?",
        f"Tôi đang phân vân giữa {pair}, nên chọn thế nào?",
        f"Nêu điểm khác biệt cốt lõi của {pair}",
        f"Giải thích {pair} và cho ví dụ dùng từng cái",
        f"{pair.capitalize()} có thể thay thế cho nhau không?",
    ])
    extra = _pick(rng, [
        "",
        " Một mẹo là ghi lại lý do chọn để sau này dễ xem lại.",
        " Nếu chưa chắc, hãy thử cả hai trên một ví dụ nhỏ trước.",
        " Đừng chọn theo thói quen mà theo yêu cầu thực tế của dự án.",
    ])
    return {"messages": [_user(q), _assistant(f"Khác biệt chính: {a}. Do đó {b}. Tốt nhất là chọn theo ngữ cảnh sử dụng của bạn.{extra}")]}


# --------------------------------------------------------------------------- code

def code_record(question: str, code: str, explain: str, rng: random.Random | None = None) -> Record:
    if rng is not None:
        question = _pick(rng, [question, f"Giúp tôi với: {question[0].lower()}{question[1:]}",
                               f"{question} — viết code ngắn gọn"])
    return {"messages": [_user(question), _assistant(f"```python\n{code}\n```\n{explain}")]}


def make_sum_range(rng: random.Random) -> Record:
    a, b = sorted((_num(rng, 1, 500), _num(rng, 1, 500)))
    ans = sum(range(a, b + 1))
    code = f"def tong_doan(a: int, b: int) -> int:\n    return sum(range(a, b + 1))\n\nprint(tong_doan({a}, {b}))"
    assert sum(range(a, b + 1)) == ans
    return code_record(
        f"Viết hàm Python tính tổng các số nguyên từ {a} đến {b}",
        code,
        f"`sum(range({a}, {b}))` cộng mọi số từ {a} đến {b}, cho kết quả **{ans}**.",
        rng)


def make_even_filter(rng: random.Random) -> Record:
    n = _num(rng, 5, 40)
    a, b = _num(rng, 1, 60), _num(rng, 61, 200)
    ans = [x for x in range(a, b) if x % 2 == 0][:n]
    code = (f"def so_chan(lo: int, hi: int, gioi_han: int) -> list[int]:\n"
            f"    return [x for x in range(lo, hi + 1) if x % 2 == 0][:gioi_han]\n\n"
            f"print(so_chan({a}, {b}, {n}))")
    return code_record(
        f"Viết hàm Python trả về {n} số chẵn đầu tiên trong đoạn từ {a} đến {b}",
        code,
        f"List comprehension lọc các số chia hết cho 2 rồi cắt lấy {n} phần tử đầu, kết quả là `{_fmt_list(ans)}`.",
        rng)


def make_factorial(rng: random.Random) -> Record:
    import math
    n = _num(rng, 1, 30)
    ans = math.factorial(n)
    code = ("import math\n\n\ndef giai_thua(n: int) -> int:\n    return math.factorial(n)\n\n"
            f"print(giai_thua({n}))")
    q = _pick(rng, [f"Viết hàm Python tính {n} giai thừa ({n}!)", f"Tính {n}! bằng Python", f"Viết code tính giai thừa của {n}"])
    return code_record(q, code, f"`math.factorial` tính tích 1 × 2 × … × {n}, kết quả **{ans}**.", rng)


def make_prime_check(rng: random.Random) -> Record:
    n = _num(rng, 2, 400)
    is_prime = n > 1 and all(n % i for i in range(2, int(n ** 0.5) + 1))
    code = ("def la_nguyen_to(n: int) -> bool:\n"
            "    if n < 2:\n        return False\n"
            "    for i in range(2, int(n ** 0.5) + 1):\n"
            "        if n % i == 0:\n            return False\n"
            "    return True\n\n"
            f"print(la_nguyen_to({n}))")
    verdict = "là số nguyên tố" if is_prime else "không phải số nguyên tố"
    q = _pick(rng, [f"Viết hàm Python kiểm tra {n} có phải số nguyên tố không",
                    f"Viết code Python xem {n} là số nguyên tố hay hợp số",
                    f"Kiểm tra tính nguyên tố của {n} bằng Python"])
    return code_record(q, code,
                       f"Chỉ chia thử đến căn bậc hai của n nên chạy nhanh. {n} {verdict}.",
        rng)


def make_string_reverse(rng: random.Random) -> Record:
    s = _pick(rng, ["xin chào", "lập trình python", "học máy", "dữ liệu lớn", "trí tuệ nhân tạo",
                    "thời tiết đẹp", "quản lý chi tiêu", "đọc sách buổi tối", "chạy bộ sáng sớm",
                    "cơ sở dữ liệu", "phân tích văn bản", "thiết kế đồ hoạ"])
    ans = s[::-1]
    code = ("def dao_chuoi(s: str) -> str:\n    return s[::-1]\n\n" f"print(dao_chuoi('{s}'))")
    q = _pick(rng, [f"Viết hàm Python đảo ngược chuỗi '{s}'", f"Đảo ngược câu '{s}' bằng Python",
                    f"Làm sao đảo ngược chuỗi '{s}' trong Python?"])
    return code_record(q, code, f"Slice `[::-1]` duyệt từ cuối về đầu, kết quả là `{ans}`.", rng)


def make_char_count(rng: random.Random) -> Record:
    s = _pick(rng, ["hôm nay trời đẹp", "python rất thú vị", "học đi đôi với hành", "kiên trì là chìa khoá",
                    "buổi sáng yên tĩnh", "đọc sách trước khi ngủ", "chạy bộ mỗi sáng", "ăn sáng đầy đủ",
                    "làm việc theo kế hoạch", "nghỉ ngơi đúng lúc", "giữ bình tĩnh khi gặp lỗi"])
    ch = rng.choice([c for c in s if c.isalpha() and c.isascii()] or ["n"])
    ans = s.lower().count(ch)
    code = ("def dem_ky_tu(s: str, ch: str) -> int:\n    return s.lower().count(ch.lower())\n\n"
            f"print(dem_ky_tu('{s}', '{ch}'))")
    q = _pick(rng, [f"Đếm số lần ký tự '{ch}' xuất hiện trong câu '{s}'",
                    f"Câu '{s}' có bao nhiêu chữ '{ch}'?",
                    f"Viết code đếm ký tự '{ch}' trong '{s}'"])
    return code_record(q, code,
                       f"Chuỗi có {ans} ký tự '{ch}' (không phân biệt hoa thường).",
        rng)


def make_sort_list(rng: random.Random) -> Record:
    xs = [rng.randint(1, 200) for _ in range(8)]
    ans = sorted(xs)
    code = f"xs = {_fmt_list(xs)}\nprint(sorted(xs))\nprint(sorted(xs, reverse=True))"
    return code_record(f"Viết code Python sắp xếp danh sách tăng dần và giảm dần: {_fmt_list(xs)}", code,
                       f"`sorted` trả list mới, không sửa list gốc khi dùng như vậy. Tăng dần: {_fmt_list(ans)}.",
        rng)


def make_maxmin(rng: random.Random) -> Record:
    xs = [rng.randint(1, 999) for _ in range(10)]
    code = f"xs = {_fmt_list(xs)}\nprint('max:', max(xs))\nprint('min:', min(xs))"
    return code_record(f"Tìm số lớn nhất và nhỏ nhất trong danh sách {_fmt_list(xs)}", code,
                       f"`max` trả {max(xs)}, `min` trả {min(xs)}. Cả hai đều chạy trong O(n).",
        rng)


def make_average(rng: random.Random) -> Record:
    xs = [rng.randint(1, 100) for _ in range(6)]
    avg = round(sum(xs) / len(xs), 2)
    code = f"xs = {_fmt_list(xs)}\nprint(round(sum(xs) / len(xs), 2))"
    return code_record(f"Tính trung bình cộng của danh sách {_fmt_list(xs)}", code,
                       f"Tổng chia cho số phần tử, kết quả khoảng **{avg}**.",
        rng)


def make_dict_from_lists(rng: random.Random) -> Record:
    keys = ["a", "b", "c", "d"][: rng.randint(2, 4)]
    vals = [rng.randint(1, 50) for _ in keys]
    keys_src = "[" + ", ".join(f"'{k}'" for k in keys) + "]"
    keys_view = "[" + ", ".join(keys) + "]"
    code = (f"keys = {keys_src}\nvals = {_fmt_list(vals)}\n"
            "print(dict(zip(keys, vals)))")
    q = _pick(rng, [f"Tạo dictionary từ hai danh sách key {keys_view} và value {_fmt_list(vals)}",
                    f"Ghép {keys_view} với {_fmt_list(vals)} thành dict trong Python"])
    return code_record(q, code,
                       f"`zip` ghép từng cặp, `dict` dựng bảng, kết quả {dict(zip(keys, vals))}.",
        rng)


def make_file_write(rng: random.Random) -> Record:
    name = _pick(rng, ["bao_cao.txt", "ghi_chu.txt", "ket_qua.txt", "todo.txt", "nhat_ky.md",
                  "ke_hoach.txt", "linh_tinh.txt"])
    text = _pick(rng, ["học xong bài 5", "chạy thử nghiệm lúc 9h", "hoàn thành 3 bài tập", "ghi lại ý tưởng mới",
                       "mai kiểm tra lại bản nháp", "đã nộp báo cáo tuần", "họp lúc 14h thứ sáu",
                       "đã kiểm tra dữ liệu", "chuẩn bị slide thuyết trình"])
    code = (f"with open('{name}', 'w', encoding='utf-8') as f:\n"
            f"    f.write('{text}')\n\n"
            f"print(open('{name}', encoding='utf-8').read())")
    q = _pick(rng, [f"Viết code Python ghi dòng '{text}' vào file {name} rồi đọc lại",
                    f"Làm sao lưu dòng '{text}' vào {name}?",
                    f"Ghi '{text}' ra {name} và in lại nội dung"])
    return code_record(q, code,
                       "Dùng `with` để file tự đóng, `encoding='utf-8'` giữ đúng tiếng Việt.",
        rng)


def make_csv_rows(rng: random.Random) -> Record:
    n = rng.randint(3, 6)
    lines = ["ten,tuoi"] + [f"nguoi{i},{rng.randint(18, 40)}" for i in range(1, n + 1)]
    code = ("import csv\n\n"
            f"data = '''{chr(10).join(lines)}'''\n"
            "rows = data.strip().splitlines()\n"
            "print(len(rows) - 1, 'dòng dữ liệu')\n"
            "for row in csv.reader(rows[1:]):\n"
            "    print(row)")
    return code_record("Viết code Python đọc nội dung CSV sau và đếm số dòng dữ liệu (bỏ header):\n" + "\n".join(lines), code,
                       f"File có **{n} dòng dữ liệu** ngoài header; `csv.reader` tách từng cột an toàn.",
        rng)


def make_json_field(rng: random.Random) -> Record:
    key = _pick(rng, ["port", "host", "name", "timeout", "debug", "region"])
    vals = {"port": ["8080", "9090", "3000"], "host": ["localhost", "127.0.0.1", "server.local"],
            "name": ["dvc-alpha", "proj-x", "ban-do"],
            "timeout": ["30", "60", "120"], "debug": ["true", "false"],
            "region": ["ap-southeast-1", "us-east-1", "eu-west-2"]}[key]
    val = _pick(rng, vals)
    code = ('import json\n\n'
            f"raw = '{{\"{key}\": \"{val}\"}}'\n"
            f"print(json.loads(raw)['{key}'])\n")
    q = _pick(rng, [f"Viết code đọc field '{key}' từ chuỗi JSON sau: {{\"{key}\": \"{val}\"}}",
                    f"Lấy giá trị '{key}' trong JSON {{\"{key}\": \"{val}\"}} bằng Python",
                    f"Parse JSON {{\"{key}\": \"{val}\"}} và in '{key}'"])
    return code_record(q, code,
                       f"`json.loads` chuyển chuỗi thành dict, sau đó lấy `['{key}']` được giá trị **{val}**.",
        rng)


def make_loop_trace(rng: random.Random) -> Record:
    n = _num(rng, 3, 40)
    mode = _pick(rng, ["square", "triple", "double", "sum"])
    acc, total = [], 0
    for i in range(1, n + 1):
        step = {"square": i * i, "triple": i * 3, "double": i * 2, "sum": i}[mode]
        total += step
        acc.append(total)
    op = {"square": "bình phương", "triple": "nhân 3", "double": "nhân 2", "sum": ""}[mode]
    body = f"    tong += i ** 2" if mode == "square" else f"    tong += i * {3 if mode == 'triple' else 2}" if mode != "sum" else "    tong += i"
    code = (f"tong = 0\nfor i in range(1, {n + 1}):\n{body}\n"
            "    print(i, tong)\nprint('Tổng:', tong)")
    q = _pick(rng, [f"Viết vòng lặp Python cộng dồn {op} từ 1 đến {n} và in từng bước",
                    f"Code Python tính tổng {op + ' của ' if op else ''}các số 1..{n}, in tổng sau mỗi bước"])
    return code_record(q, code,
                       f"Kết quả cộng dồn: {_fmt_list(acc[:6])}{' …' if len(acc) > 6 else ''}. Tổng cuối cùng là **{total}**.",
        rng)


def make_exception(rng: random.Random) -> Record:
    a, b = _num(rng, 1, 100), rng.choice([0, 0, _num(rng, 1, 10)])
    if b == 0:
        code = (f"try:\n    print({a} / {b})\nexcept ZeroDivisionError:\n    print('Không thể chia cho 0')")
        explain = "Chia cho 0 ném `ZeroDivisionError`, khối `except` bắt lỗi và in thông báo thay vì làm sập chương trình."
    else:
        code = (f"try:\n    print(int('{a}') / {b})\nexcept (ValueError, ZeroDivisionError) as e:\n"
                "    print('Lỗi:', e)")
        explain = f"Bắt cả `ValueError` lẫn `ZeroDivisionError`, bắt đầu từ lớp con rồi đến lớp cha. {a} / {b} = {a / b:.4f}."
    return code_record(f"Viết code Python xử lý lỗi khi thực hiện phép chia {a} / {b}", code, explain, rng)


def make_class(rng: random.Random) -> Record:
    name, fields = _pick(rng, [
        ("HinhChuNhat", ("dai", "rong")),
        ("TaiKhoan", ("chu_tai_khoan", "so_du")),
        ("Diem2D", ("x", "y")),
        ("SanPham", ("ten", "gia")),
        ("NhanVien", ("ho_ten", "luong")),
        ("SinhVien", ("ma_sv", "diem_tb")),
        ("XeHoi", ("hang", "nam_san_xuat")),
        ("DonHang", ("ma_don", "tong_tien")),
    ])
    f0, f1 = fields
    v0, v1 = (_pick(rng, ["An", "Bình", "Chi", "Dũng"]), _num(rng, 10, 99)) if f0 in ("ten", "ho_ten", "chu_tai_khoan") else (_num(rng, 1, 20), _num(rng, 1, 20))
    body = (f"class {name}:\n    def __init__(self, {f0}, {f1}):\n"
            f"        self.{f0} = {f0}\n        self.{f1} = {f1}\n\n"
            "    def mo_ta(self) -> str:\n"
            f"        return f'{name}({{self.{f0}!r}}, {{self.{f1}!r}})'\n\n"
            f"print({name}({v0!r}, {v1!r}).mo_ta())")
    q = _pick(rng, [f"Viết class Python tên {name} có hai thuộc tính {f0} và {f1}, kèm phương thức mô tả",
                    f"Tạo class {name}({f0}, {f1}) trong Python và in thông tin object"])
    return code_record(q, body,
                       "`__init__` nhận tham số khi tạo object, `self` giữ trạng thái riêng của từng object.",
        rng)


def make_recursion_fib(rng: random.Random) -> Record:
    n = rng.randint(5, 20)
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    code = ("def fib(n: int) -> int:\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n\n"
            f"print(fib({n}))")
    return code_record(f"In số Fibonacci thứ {n} bằng Python", code,
                       f"Bản lặp tránh đệ quy nên không tràn stack; Fibonacci thứ {n} là **{a}**.",
        rng)


def make_comprehension(rng: random.Random) -> Record:
    k = rng.randint(2, 30)
    kind = rng.choice(["odd_square", "even", "div3"])
    if kind == "odd_square":
        ans = [x ** 2 for x in range(k + 1) if x % 2 == 1]
        desc, filt = "bình phương các số lẻ", "x % 2 == 1"
        expr = "x ** 2"
    elif kind == "even":
        ans = [x for x in range(k + 1) if x % 2 == 0]
        desc, filt = "các số chẵn", "x % 2 == 0"
        expr = "x"
    else:
        ans = [x * 10 for x in range(k + 1) if x % 3 == 0]
        desc, filt = "nhân 10 các số chia hết cho 3", "x % 3 == 0"
        expr = "x * 10"
    code = f"print([{expr} for x in range({k + 1}) if {filt}])"
    q = _pick(rng, [f"Viết list comprehension trả về {desc} từ 0 đến {k}",
                    f"Dùng comprehension lấy {desc} trong khoảng 0..{k}",
                    f"Viết một dòng Python tính {desc} từ 0 đến {k}"])
    return code_record(q, code, f"Điều kiện `{filt}` giữ đúng phần tử, kết quả {_fmt_list(ans)}.", rng)


def make_read_file_python(rng: random.Random) -> Record:
    n = rng.randint(2, 20)
    base = rng.randint(1, 50)
    body = "\n".join(f"x{i} = {base + i}" for i in range(1, n + 1))
    code = (f"code = '''{body}'''\n"
            "ns = {}\n"
            "exec(code, ns)\n"
            'print([ns[f"x{j}"] for j in range(1, %d)])' % (n + 1))
    q = _pick(rng, ["Đoạn code sau được lưu trong file, viết code Python chạy nội dung đó:\n",
                    "File script.py chứa đoạn code dưới đây. Viết code đọc file rồi chạy nội dung nó:\n"])
    return code_record(q + body, code,
                       "`exec` chạy chuỗi code trong namespace riêng; để đọc lại biến ta truyền dict `ns` và lấy giá trị từ đó.",
        rng)


# --------------------------------------------------------------------------- tool_call

def tool_record(user: str, steps: list[tuple[str, dict, str]], final: str) -> Record:
    msgs = [_system_tool(), _user(user)]
    for assistant_text, (name, args, tool_result) in steps:
        msgs.append(_assistant((assistant_text + "\n" if assistant_text else "") + tool_call_block(name, args)))
        msgs.append({"role": "tool", "content": tool_result})
    msgs.append(_assistant(final))
    return {"messages": msgs}


def make_tool_read(rng: random.Random) -> Record:
    f = _pick(rng, FILES)
    sec = _pick(rng, SECTIONS)
    path = f if sec in (".", "workspace") else f"{sec}/{f}"
    content = _pick(rng, ["cổng mặc định là 8080", "hàm main in ra lời chào", "danh sách công việc tuần này",
                          "phiên bản 1.4.2", "tài khoản chưa kích hoạt"])
    q = _pick(rng, [f"Đọc file {path} và cho tôi biết nội dung",
                    f"File {path} có nội dung gì?",
                    f"Mở {path} giúp tôi",
                    f"Cho tôi xem {path}"])
    return tool_record(q,
                       [("Tôi sẽ đọc file đó.", ("read_file", {"path": path}, content))],
                       f"Nội dung `{path}`: {content}.")


def make_tool_write(rng: random.Random) -> Record:
    name = _pick(rng, ["ghi_chu.txt", "ket_qua.txt", "todo.txt", "nhat_ky.md", "bai_tap.py", "danh_sach.txt", "loi_nhan.txt"])
    content = _pick(rng, ["hoàn thành bài 3 trước tối", "kết quả: 42", "họp lúc 9h thứ hai", "print('xin chào')",
                          "mua sữa và bánh mì", "đã kiểm tra lần hai", "gọi cho chị Lan chiều nay"])
    q = _pick(rng, [f"Ghi nội dung '{content}' vào file {name}",
                    f"Lưu dòng '{content}' ra {name} giúp tôi",
                    f"Tạo file {name} chứa '{content}'"])
    return tool_record(q,
                       [("Tôi sẽ ghi file cho bạn.", ("write_file", {"path": name, "content": content}, f"Đã ghi {len(content)} ký tự vào {name}"))],
                       f"Đã lưu vào `{name}`.")


def make_tool_list(rng: random.Random) -> Record:
    sec = _pick(rng, SECTIONS)
    names = rng.sample(FILES, rng.randint(2, 5))
    listed = "\n".join(sorted(names))
    q = _pick(rng, [f"Liệt kê các file trong thư mục {sec}",
                    f"Thư mục {sec} có những gì?",
                    f"Xem nhanh nội dung thư mục {sec}",
                    f"Cho tôi danh sách file ở {sec}"])
    return tool_record(q,
                       [("Tôi sẽ xem thư mục.", ("list_dir", {"path": sec}, listed))],
                       f"Thư mục `{sec}` có {len(names)} mục: " + ", ".join(f"`{n}`" for n in sorted(names)) + ".")


def make_tool_run_python(rng: random.Random) -> Record:
    a, b = _num(rng, 2, 99), _num(rng, 2, 99)
    code = f"print({a} * {b})"
    return tool_record(f"Tính {a} nhân {b} bằng Python",
                       [("Tôi sẽ chạy Python.", ("run_python", {"code": code}, str(a * b)))],
                       f"{a} × {b} = **{a * b}**.")


def make_tool_multi(rng: random.Random) -> Record:
    f = _pick(rng, FILES)
    words = _pick(rng, ["xin chào", "dữ liệu thử", "học python", "ghi chú nhanh"])
    n = len(words.split())
    q = _pick(rng, [f"Đọc {f}, đếm số từ rồi ghi kết quả vào so_tu.txt",
                    f"Cho tôi số từ của {f} và lưu vào so_tu.txt",
                    f"Kiểm tra {f} có bao nhiêu từ, ghi ra so_tu.txt giúp tôi"])
    return tool_record(q,
                       [("Tôi sẽ đọc file trước.", ("read_file", {"path": f}, words)),
                        ("Đã có nội dung, tôi sẽ chạy code đếm từ.", ("run_python", {"code": f"print(len('{words}'.split()))"}, str(n))),
                        ("Đang ghi kết quả.", ("write_file", {"path": "so_tu.txt", "content": f"Số từ: {n}"}, "Đã ghi 9 ký tự vào so_tu.txt"))],
                       f"`{f}` có **{n} từ**, tôi đã lưu vào `so_tu.txt`.")


def make_tool_error(rng: random.Random) -> Record:
    f = _pick(rng, ["khong_ton_tai.txt", "thieu.json", "chua_co.py", "config_cu.yaml", "du_lieu_2023.csv",
                    "main_old.py", "secret.key", "backup.zip"])
    kind = rng.randrange(3)
    if kind < 2:
        return tool_record(_pick(rng, [f"Đọc file {f} giúp tôi", f"Cho tôi xem nội dung {f}", f"{f} có gì trong đó?"]),
                           [("Tôi sẽ đọc file đó.", ("read_file", {"path": f}, f"LỖI: Không tìm thấy file: {f}"))],
                           f"File `{f}` không có trong workspace. Bạn kiểm tra lại tên file hoặc liệt kê thư mục để tìm đúng tên giúp tôi.")
    return tool_record(_pick(rng, [f"Ghi dòng 'tạm' vào ../../{f}", f"Ghi chú vào ../../{f} giúp tôi"]),
                       [("Tôi sẽ thử ghi file đó.", ("write_file", {"path": f"../../{f}", "content": "tạm"},
                                                     f"LỖI: Đường dẫn nằm ngoài workspace: ../../{f}"))],
                       f"Không ghi được: đường dẫn `../../{f}` nằm ngoài workspace. Tôi chỉ thao tác được với file trong workspace; hãy dùng đường dẫn tương đối bên trong workspace.")


def make_tool_no_call(rng: random.Random) -> Record:
    q, ans = _pick(rng, [
        ("Giải thích vòng lặp for trong Python", "Vòng lặp `for` duyệt lần lượt từng phần tử của một iterable như list, tuple hay range, và dừng khi hết phần tử."),
        ("Tuple khác list ở đâu?", "`tuple` không thay đổi được sau khi tạo, còn `list` thì có. Tuple phù hợp cho dữ liệu cố định."),
        ("Vì sao nên viết test cho code?", "Test giúp phát hiện lỗi khi sửa code về sau và làm tài liệu sống cho hành vi mong đợi."),
        ("Global và local variable khác gì nhau?", "Biến toàn cục khai báo ngoài hàm, ai cũng đọc được; biến cục bộ chỉ sống trong hàm tạo ra nó."),
        ("Dict comprehension là gì?", "Dict comprehension tạo dict bằng một biểu thức, ví dụ `{x: x*x for x in range(5)}` cho ra bảng bình phương."),
        ("Khi nào nên dùng set thay cho list?", "Dùng `set` khi cần kiểm tra phần tử đã có hay chưa, vì tra cứu trung bình là O(1) và tự loại trùng."),
        ("Viết docstring cho hàm thế nào là đúng?", "Đặt docstring ngay dòng đầu của hàm, trong ba dấu nháy kép, mô tả ngắn mục đích rồi tới tham số và giá trị trả về."),
        ("Code Python nên thụt lề mấy dấu cách?", "Theo style chuẩn PEP 8 là 4 dấu cách cho mỗi cấp, không trộn tab với dấu cách."),
        ("`yield` dùng để làm gì?", "`yield` biến hàm thành generator: hàm trả từng giá trị một khi được gọi lặp, không tạo cả list trong bộ nhớ."),
        ("Nên đặt tên biến bằng tiếng Anh hay tiếng Việt?", "Nên thống nhất trong cả dự án; nhiều nhóm chọn tiếng Anh cho tên biến và giữ tiếng Việt cho nội dung hiển thị với người dùng."),
        ("Vì sao `==` khác `is`?", "`==` so sánh giá trị, `is` so sánh cùng một object trong bộ nhớ; chỉ dùng `is` với `None`, `True`, `False`."),
        ("List và generator khác nhau thế nào?", "List tạo toàn bộ phần tử ngay, generator sinh dần khi lặp nên tiết kiệm bộ nhớ với dữ liệu lớn."),
        ("Khi nào nên chia code thành nhiều module?", "Khi một file khó theo dõi hoặc có nhóm hàm dùng chung; chia theo trách nhiệm giúp tái sử dụng và test dễ hơn."),
    ])
    return {"messages": [_system_tool(), _user(_pick(rng, [
        q, f"Tôi đang học Python, {q[0].lower()}{q[1:]}",
        f"Giải thích giúp tôi: {q[0].lower()}{q[1:]}",
    ])), _assistant(ans)]}


CHAT_TEMPLATES = [
    Template("place", 60, make_place_fact),
    Template("dish", 60, make_dish),
    Template("concept", 60, make_concept),
    Template("habit", 60, make_habit),
    Template("arithmetic", 100, make_arithmetic),
    Template("math_explain", 40, make_math_explain),
    Template("conversion", 60, make_conversion),
    Template("long_form", 40, make_long_form),
    Template("compare", 40, make_compare),
]

CODE_TEMPLATES = [
    Template("sum_range", 40, make_sum_range),
    Template("even_filter", 25, make_even_filter),
    Template("factorial", 25, make_factorial),
    Template("prime", 25, make_prime_check),
    Template("reverse", 15, make_string_reverse),
    Template("char_count", 20, make_char_count),
    Template("sort", 25, make_sort_list),
    Template("maxmin", 25, make_maxmin),
    Template("average", 20, make_average),
    Template("dict_lists", 20, make_dict_from_lists),
    Template("file_write", 20, make_file_write),
    Template("csv_rows", 20, make_csv_rows),
    Template("json_field", 20, make_json_field),
    Template("loop_trace", 20, make_loop_trace),
    Template("exception", 20, make_exception),
    Template("class", 20, make_class),
    Template("fib", 15, make_recursion_fib),
    Template("comprehension", 15, make_comprehension),
    Template("read_file", 15, make_read_file_python),
]

TOOL_TEMPLATES = [
    Template("read", 40, make_tool_read),
    Template("write", 30, make_tool_write),
    Template("list", 30, make_tool_list),
    Template("run_python", 30, make_tool_run_python),
    Template("multi", 25, make_tool_multi),
    Template("error", 15, make_tool_error),
    Template("no_call", 20, make_tool_no_call),
]


def _question_key(rec: Record) -> str:
    for m in rec["messages"]:
        if m["role"] == "user":
            return m["content"].strip().lower()
    return ""


def raw_question_keys(raw_dir: Path) -> set[str]:
    keys = set()
    for path in sorted(raw_dir.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                keys.add(_question_key(json.loads(line)))
    return keys


def generate(templates: list[Template], seed: int, skip: set[str] | None = None,
             max_attempts_per_sample: int = 50) -> list[Record]:
    rng = random.Random(seed)
    skip = skip or set()
    out: list[Record] = []
    for tpl in templates:
        made = 0
        attempts = 0
        while made < tpl.count:
            if attempts > tpl.count * max_attempts_per_sample:
                raise RuntimeError(f"Khuôn '{tpl.name}' không sinh đủ mẫu phân biệt: {made}/{tpl.count}")
            attempts += 1
            rec = tpl.make(rng)
            key = json.dumps(rec, ensure_ascii=False, sort_keys=True)
            if key in tpl.seen or _question_key(rec) in skip:
                continue
            tpl.seen.add(key)
            out.append(rec)
            made += 1
    rng.shuffle(out)
    return out


CODE_FENCE = re.compile(r"```python\n(.*?)\n```", re.DOTALL)


def verify_code(record: Record, timeout_s: int = 10) -> list[str]:
    """Chạy đoạn code trong câu trả lời và đối chiếu với giải thích. Trả danh sách lỗi."""
    match = CODE_FENCE.search(record["messages"][-1]["content"])
    if not match:
        return ["không có khối ```python```"]
    code = match.group(1)
    with tempfile.TemporaryDirectory() as workdir:
        try:
            proc = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=timeout_s, cwd=workdir, env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            )
        except subprocess.TimeoutExpired:
            return ["quá thời gian"]
    stdout = proc.stdout
    if proc.returncode != 0:
        return [f"lỗi khi chạy: {proc.stderr.strip().splitlines()[-1][:120]}"]
    numbers = re.findall(r"\*\*(-?[\d.,]+)\*\*", record["messages"][-1]["content"])
    missing = [n for n in numbers if n not in stdout and n.replace(",", "") not in stdout]
    return [f"kết quả {n} không có trong đầu ra" for n in missing]


def verify_all(data: dict[str, list[Record]], verbose: bool = True) -> int:
    """Kiểm chứng mọi mẫu code đã sinh. Trả số mẫu lỗi."""
    failed = 0
    for i, rec in enumerate(data["code_vi"]):
        errors = verify_code(rec)
        if errors:
            failed += 1
            if verbose and failed <= 10:
                print(f"  LỖI code_vi[{i}]: {'; '.join(errors)}")
                print(f"    câu hỏi: {rec['messages'][0]['content'][:80]}")
    if verbose:
        print(f"Kiểm chứng code: {len(data['code_vi']) - failed}/{len(data['code_vi'])} mẫu chạy đúng")
    return failed


def write_jsonl(path: Path, rows: list[Record]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build(seed: int = 2026, scale: float = 1.0, raw_dir: str | Path = "data/raw") -> dict[str, list[Record]]:
    def scaled(templates):
        return [Template(t.name, max(1, int(t.count * scale)), t.make) for t in templates]

    skip = raw_question_keys(Path(raw_dir))
    data = {
        "chat_vi": generate(scaled(CHAT_TEMPLATES), seed, skip),
        "code_vi": generate(scaled(CODE_TEMPLATES), seed + 1, skip),
        "tool_call": generate(scaled(TOOL_TEMPLATES), seed + 2, skip),
    }
    failed = verify_all(data)
    if failed:
        raise RuntimeError(f"{failed} mẫu code chạy sai; sửa khuôn trước khi sinh tiếp")
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data/generated")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--scale", type=float, default=1.0, help="nhân số mẫu, ví dụ 1.5")
    args = parser.parse_args()

    data = build(args.seed, args.scale)
    out = Path(args.out_dir)
    for name, rows in data.items():
        write_jsonl(out / f"{name}.jsonl", rows)
    total = sum(len(r) for r in data.values())
    print(" | ".join(f"{k}: {len(v)}" for k, v in data.items()) + f" | tổng: {total}")
    print(f"Đã ghi vào {out}")


if __name__ == "__main__":
    main()
