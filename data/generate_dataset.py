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
from functools import lru_cache
from pathlib import Path
from typing import Callable

from .tool_system_prompt import TOOL_SYSTEM_PROMPT

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
    return {"role": "system", "content": TOOL_SYSTEM_PROMPT}



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


def make_tool_search_files(rng: random.Random) -> Record:
    folder = _pick(rng, ["app", "tests", "docs", "scripts"])
    pattern = _pick(rng, ["*.py", "*.md", "*.ps1"])
    query = _pick(rng, [f"Tìm các file {pattern} trong {folder}", f"Liệt kê file theo mẫu {pattern} ở {folder}"])
    return tool_record(query,
        [("Tôi sẽ tìm file theo mẫu.", ("search_files", {"pattern": pattern, "path": folder}, f"{folder}/main.py\n{folder}/utils.py"))],
        f"Đã tìm thấy các file phù hợp với `{pattern}` trong `{folder}`.")


def make_tool_search_code(rng: random.Random) -> Record:
    token = _pick(rng, ["def main", "class Agent", "TODO", "route("])
    query = f"Tìm `{token}` trong mã nguồn project"
    return tool_record(query,
        [("Tôi sẽ tìm trong source code.", ("search_code", {"query": token}, f"app/main.py:12: {token}"))],
        f"Đã tìm các vị trí chứa `{token}` trong source code.")


def make_tool_project_tree(rng: random.Random) -> Record:
    return tool_record("Cho tôi xem cấu trúc project hiện tại",
        [("Tôi sẽ quét cấu trúc project.", ("project_tree", {}, "app/\n tests/\n docs/\n README.md"))],
        "Project có các thư mục chính `app`, `tests`, `docs` và file `README.md`.")


def make_tool_preview(rng: random.Random) -> Record:
    content = "def hello():\n    return 'Xin chào'\n"
    return tool_record("Xem trước nội dung mới cho file hello.py, chưa ghi file",
        [("Tôi sẽ tạo diff để bạn xem trước.", ("preview_file_change", {"path": "hello.py", "new_content": content}, "--- /dev/null\n+++ b/hello.py\n+def hello():\n+    return 'Xin chào'"))],
        "Đây là bản xem trước; file chưa được ghi. Hãy xác nhận trước khi áp dụng.")


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


# --------------------------------------------------------------------------- chat: hội thoại nhiều lượt

MULTITURN = [
    {
        "first_q": ["Tôi mới bắt đầu học Python, nên làm gì trước?",
                    "Cách học Python cho người mới là gì?"],
        "first_a": "Bắt đầu từ cú pháp cơ bản: biến, kiểu dữ liệu, câu lệnh điều kiện và vòng lặp. Sau đó làm một loạt bài tập nhỏ như tính tổng, đảo chuỗi, đếm ký tự để tay quen cú pháp. Đừng đọc hết sách trước khi gõ code.",
        "followups": [
            ("Cho tôi ví dụ bài tập đầu tiên", "Thử bài này: nhập một danh sách số, in ra số lớn nhất và số nhỏ nhất.\n\n```python\nxs = [4, 9, 1, 7]\nprint(max(xs), min(xs))\n```\nChạy được rồi thì tự viết lại mà không nhìn code mẫu."),
            ("Mỗi ngày nên học bao lâu?", "Khoảng 45 đến 60 phút là đủ nếu học đều. Chia thành 20 phút đọc và 30 phút gõ code; đừng học dồn 5 tiếng vào cuối tuần vì sẽ quên nhanh hơn."),
            ("Khi nào thì nên học framework?", "Khi đã viết được hàm và class cơ bản mà không phải tra cú pháp. Học framework quá sớm thường dẫn tới copy code mà không hiểu lỗi."),
        ],
    },
    {
        "first_q": ["Tôi muốn giảm cân, nên bắt đầu từ đâu?",
                    "Giảm cân an toàn thì làm thế nào?"],
        "first_a": "Nguyên tắc chính là ăn ít calo hơn mức tiêu hao, nhưng giảm từ từ: khoảng 300 đến 500 calo mỗi ngày. Tăng vận động, ưu tiên đạm và rau, hạn chế nước ngọt. Không bỏ bữa vì dễ ăn bù sau đó.",
        "followups": [
            ("Có cần tập gym không?", "Không bắt buộc. Đi bộ nhanh 30 phút mỗi ngày đã tạo thâm hụt calo đáng kể. Tập kháng lực thì giúp giữ cơ khi giảm cân, nhưng có thể bắt đầu bằng bài tập với trọng lượng cơ thể."),
            ("Bao lâu thì thấy kết quả?", "Thường 0,3 đến 0,5 kg mỗi tuần là mức bền vững. Giảm nhanh hơn thường mất nước và cơ chứ không phải mỡ, dễ tăng lại."),
            ("Tôi hay ăn đêm thì sửa thế nào?", "Đặt bữa tối đủ đạm và chất xơ để no lâu, uống nước khi thấy thèm, và đánh răng sau bữa tối. Nếu vẫn đói thì chuẩn bị sẵn sữa chua không đường thay vì đồ ngọt."),
        ],
    },
    {
        "first_q": ["Làm sao để nói tiếng Anh tự tin hơn?",
                    "Tôi sợ nói tiếng Anh, cải thiện thế nào?"],
        "first_a": "Nỗi sợ chủ yếu đến từ việc sợ sai. Hãy luyện nói một mình trước: ghi âm 1 phút mỗi ngày về một chủ đề quen. Sau đó nói với người thật trong tình huống ít áp lực như lớp học hoặc nhóm nhỏ.",
        "followups": [
            ("Nên học ngữ pháp hay luyện nói trước?", "Luyện nói trước nhưng có khung cơ bản. Nắm thì hiện tại, quá khứ, tương lai là đủ để nói chuyện hằng ngày; các thì phức tạp học sau khi đã nói được."),
            ("Có cách nào luyện một mình không?", "Có: chọn một chủ đề, đặt 3 câu hỏi, tự trả lời và ghi âm rồi nghe lại. Hoặc đọc to theo một đoạn ngắn 2 phút, chú ý nối âm."),
            ("Bao lâu thì nói được trôi chảy?", "Với 20 phút mỗi ngày, khoảng 3 đến 6 tháng sẽ thấy khác rõ ở các chủ đề quen thuộc. Mốc trôi chảy hoàn toàn thì phụ thuộc mức độ sử dụng thực tế."),
        ],
    },
    {
        "first_q": ["Tôi muốn tiết kiệm tiền nhưng lương thấp, làm sao?",
                    "Thu nhập thấp thì tiết kiệm thế nào?"],
        "first_a": "Bắt đầu bằng việc ghi lại mọi khoản chi trong 2 tuần để biết tiền đi đâu. Sau đó trích một khoản nhỏ ngay khi nhận lương, kể cả 5%, và tách sang tài khoản riêng để không tiêu.",
        "followups": [
            ("Nên cắt khoản nào trước?", "Cắt các khoản định kỳ không dùng: gói thuê bao, ăn ngoài, cà phê mang đi. Đây là khoản giảm được ngay mà không ảnh hưởng chất lượng sống nhiều."),
            ("Có nên vay để đầu tư không?", "Không nên khi chưa có quỹ dự phòng 3 đến 6 tháng chi phí. Vay đầu tư có thể lỗ kép: mất tiền đầu tư và vẫn phải trả lãi."),
            ("Quỹ dự phòng để ở đâu?", "Để ở nơi rút được trong 1 đến 2 ngày: tài khoản tiết kiệm không kỳ hạn hoặc kỳ hạn ngắn. Ưu tiên rút nhanh hơn là lãi cao."),
        ],
    },
    {
        "first_q": ["Làm sao để ngủ ngon hơn?",
                    "Tôi khó ngủ, có cách nào cải thiện?"],
        "first_a": "Giữ giờ ngủ và giờ thức cố định kể cả cuối tuần, tắt màn hình 1 giờ trước khi ngủ, và giữ phòng tối, mát. Cà phê nên ngừng sau 2 giờ chiều vì caffeine tồn tại nhiều giờ.",
        "followups": [
            ("Nằm mãi không ngủ được thì làm gì?", "Nếu sau 20 phút vẫn thức, hãy ra khỏi giường, ngồi ở ánh sáng mờ và đọc vài trang giấy cho đến khi buồn ngủ. Nằm trằn trọc trên giường làm não gắn giường với việc lo lắng."),
            ("Ngủ trưa có ảnh hưởng không?", "Có, nếu ngủ trưa quá dài hoặc quá muộn. Giữ giấc trưa 15 đến 20 phút và trước 3 giờ chiều."),
            ("Có nên dùng thuốc ngủ không?", "Chỉ khi bác sĩ kê. Thuốc ngủ dùng lâu có thể gây lệ thuộc; cách trị mất ngủ mạn tính hiệu quả nhất vẫn là liệu pháp hành vi do chuyên gia hướng dẫn."),
        ],
    },
]

# --------------------------------------------------------------------------- chat: hành vi trợ lý

FORMAT_TASKS = [
    ("gạch đầu dòng", lambda items: "\n".join(f"- {x}" for x in items),
     ["Vì sao nên viết test cho code?", "Lợi ích của việc đọc sách mỗi ngày?",
      "Vì sao nên dùng biến có tên rõ nghĩa?", "Lợi ích của việc ghi chú khi học?"]),
    ("bảng", lambda items: "| Điểm | Ý chính |\n|---|---|\n" + "\n".join(f"| {i + 1} | {x} |" for i, x in enumerate(items)),
     ["So sánh list và tuple", "So sánh HTTP và HTTPS", "So sánh SQL và NoSQL"]),
]

ONE_LINERS = [
    ("Thủ đô của Việt Nam là gì?", "Hà Nội."),
    ("Python ra đời năm nào?", "Python được công bố lần đầu năm 1991 bởi Guido van Rossum."),
    ("HTTP là viết tắt của gì?", "HTTP là HyperText Transfer Protocol."),
    ("1 km bằng bao nhiêu mét?", "1 km bằng 1.000 mét."),
    ("HTML dùng để làm gì?", "HTML dùng để mô tả cấu trúc nội dung của trang web."),
]

CLARIFY = [
    ("Đọc file giúp tôi", "Bạn muốn đọc file nào? Cho tôi tên file hoặc đường dẫn tương đối trong workspace để tôi mở đúng file."),
    ("Sửa code này cho tôi", "Bạn chưa gửi đoạn code. Hãy dán code và mô tả lỗi hoặc kết quả mong muốn, tôi sẽ chỉ ra chỗ cần sửa."),
    ("Chạy thử giúp tôi", "Bạn muốn chạy đoạn code hay file nào? Gửi nội dung code hoặc tên file trong workspace nhé."),
    ("Dịch câu này", "Bạn muốn dịch câu nào và sang ngôn ngữ gì? Gửi nội dung cần dịch giúp tôi."),
    ("Tóm tắt giúp tôi", "Bạn muốn tóm tắt văn bản nào và dài khoảng bao nhiêu? Dán nội dung hoặc cho tên file trong workspace."),
    ("Tối ưu đoạn này", "Đoạn code cần tối ưu chưa có trong tin nhắn. Bạn dán code và cho biết đang chậm ở đâu hoặc muốn tối ưu về gì nhé."),
]

UNKNOWN = [
    ("Giá vàng ngày mai là bao nhiêu?", "Tôi không biết được giá vàng ngày mai vì không có dữ liệu thời gian thực. Bạn kiểm tra trên trang của đơn vị kinh doanh vàng để có giá chính xác."),
    ("Thời tiết Hà Nội tuần sau thế nào?", "Tôi không có dữ liệu dự báo thời tiết. Bạn xem trang dự báo khí tượng để biết dự báo tuần sau."),
    ("Số điện thoại của tôi là gì?", "Tôi không biết số điện thoại của bạn: tôi không truy cập được thông tin cá nhân. Bạn kiểm tra trong cài đặt điện thoại hoặc hợp đồng dịch vụ."),
    ("Tỷ giá đô la hôm nay bao nhiêu?", "Tôi không có tỷ giá theo thời gian thực. Bạn xem bảng tỷ giá của ngân hàng để có con số hôm nay."),
    ("Mật khẩu wifi nhà tôi là gì?", "Tôi không biết mật khẩu wifi của bạn. Mật khẩu thường in trên mặt sau router hoặc bạn hỏi người quản lý mạng."),
]

REWRITES = [
    ("Chúng tôi đã tiến hành kiểm tra và nhận thấy rằng hệ thống có thể gặp sự cố trong trường hợp tải tăng cao.",
     "Ngắn gọn", "Hệ thống có thể gặp sự cố khi tải tăng cao."),
    ("Vì vậy, để đảm bảo tính ổn định, chúng ta cần thực hiện việc giám sát thường xuyên.",
     "Trang trọng hơn", "Do đó, nhằm bảo đảm tính ổn định, cần tiến hành giám sát thường xuyên."),
    ("Cái này hơi khó hiểu nên mình chưa làm được.",
     "Trang trọng hơn", "Nội dung này khá phức tạp nên tôi chưa hoàn thành được."),
    ("Mai họp lúc 9 giờ ở phòng 3, nhớ mang theo tài liệu.",
     "Ngắn gọn", "Họp 9h mai, phòng 3, mang tài liệu."),
    ("Báo cáo này trình bày kết quả khảo sát 200 người dùng trong tháng 10 và các đề xuất cải thiện.",
     "Một câu", "Báo cáo tóm tắt khảo sát 200 người dùng tháng 10 và đề xuất cải thiện."),
]

TRANSLATE = [
    ("Tôi đang học lập trình Python.", "I am learning Python programming."),
    ("Máy chủ không phản hồi trong 30 giây.", "The server did not respond for 30 seconds."),
    ("Vui lòng kiểm tra lại đường dẫn tệp.", "Please check the file path again."),
    ("Chúng tôi sẽ triển khai bản cập nhật vào thứ Sáu.", "We will deploy the update on Friday."),
    ("Hãy chạy thử nghiệm trước khi phát hành.", "Run the test before releasing."),
    ("Dữ liệu được sao lưu mỗi đêm.", "The data is backed up every night."),
    ("Hàm này trả về danh sách các số chẵn.", "This function returns a list of even numbers."),
    ("Tôi cần thêm thời gian để hoàn thành báo cáo.", "I need more time to finish the report."),
]

IDIOMS = [
    ("nước đến chân mới nhảy", "ý chỉ người chỉ làm việc khi đã sát hạn, thường dẫn tới kết quả kém."),
    ("ăn quả nhớ kẻ trồng cây", "nhắc nhớ biết ơn người đã tạo ra thành quả mình đang hưởng."),
    ("chậm mà chắc", "làm chậm nhưng kỹ, chắc chắn, thay vì nhanh mà hỏng."),
    ("có công mài sắt có ngày nên kim", "kiên trì thì việc khó cũng hoàn thành."),
    ("đi một ngày đàng học một sàng khôn", "đi nhiều thì học được nhiều điều thực tế."),
    ("gần mực thì đen gần đèn thì sáng", "môi trường xung quanh ảnh hưởng tới tính cách và hành vi."),
    ("một nắng hai sương", "chỉ sự vất vả, làm lụng từ sớm tới tối."),
    ("uống nước nhớ nguồn", "nhắc nhớ biết ơn nguồn cội, người đi trước."),
]


def make_multiturn(rng: random.Random) -> Record:
    topic = _pick(rng, MULTITURN)
    first_q = _pick(rng, topic["first_q"])
    follow_q, follow_a = _pick(rng, topic["followups"])
    msgs = [_user(first_q), _assistant(topic["first_a"]), _user(follow_q), _assistant(follow_a)]
    if rng.random() < 0.5:
        second_q, second_a = _pick(rng, [f for f in topic["followups"] if f[0] != follow_q])
        msgs += [_user(second_q), _assistant(second_a)]
    return {"messages": msgs}


def make_format_follow(rng: random.Random) -> Record:
    kind = _pick(rng, ["bullet", "one_line", "one_line"])
    if kind == "bullet":
        style, formatter, pool = FORMAT_TASKS[rng.randrange(2)]
        q0 = _pick(rng, pool)
        points = _pick(rng, [
            ["Phát hiện lỗi sớm", "Giữ hành vi cũ không bị đổi", "Tự tin sửa code", "Làm tài liệu cho người sau"],
            ["Mở rộng vốn từ", "Tăng khả năng tập trung", "Giảm căng thẳng", "Hiểu thêm nhiều góc nhìn"],
            ["Dễ đọc hơn", "Ít lỗi khi sửa", "Dễ tìm chỗ sai", "Người khác tiếp nhận nhanh"],
        ])
        q = _pick(rng, [f"{q0} Trả lời dạng {style}.", f"{q0} Trình bày dạng {style}.",
                        f"{q0} Hãy trả lời dưới dạng {style}."])
        body = formatter(points)
        assert (body.startswith("- ") or body.startswith("|")), "định dạng sai"
        return {"messages": [_user(q), _assistant(body)]}
    q0, ans = _pick(rng, ONE_LINERS)
    q = _pick(rng, [f"{q0} Trả lời đúng một câu.", f"{q0} Chỉ cần một câu.",
                    f"{q0} Trả lời gọn trong một câu."])
    assert ans.count(".") <= 2
    return {"messages": [_user(q), _assistant(ans)]}


def _ask_variants(rng: random.Random, q: str, wrappers: list[str]) -> str:
    return _pick(rng, [w.format(q=q, Q=q[0].upper() + q[1:]) for w in wrappers])


ASK_PLAIN = ["{q}", "{q}", "Cho tôi hỏi: {q}", "Giúp tôi với: {q}"]
ASK_POLITE = ["{q}", "{q}", "Bạn giải đáp giúp: {q}", "Hỏi nhanh: {q}"]


def make_clarify(rng: random.Random) -> Record:
    q, a = _pick(rng, CLARIFY)
    return {"messages": [_user(_ask_variants(rng, q, ASK_PLAIN)), _assistant(a)]}


def make_unknown(rng: random.Random) -> Record:
    q, a = _pick(rng, UNKNOWN)
    return {"messages": [_user(_ask_variants(rng, q, ASK_POLITE)), _assistant(a)]}


def make_rewrite(rng: random.Random) -> Record:
    src, mode, out = _pick(rng, REWRITES)
    q = _pick(rng, [f"Viết lại câu sau cho {mode.lower()}: \"{src}\"",
                    f"Viết lại {mode.lower()}: {src}",
                    f"Rút gọn câu này: {src}" if mode != "Trang trọng hơn" else f"Đổi câu này sang văn phong trang trọng: {src}"])
    return {"messages": [_user(q), _assistant(out)]}


def make_translate(rng: random.Random) -> Record:
    vi, en = _pick(rng, TRANSLATE)
    if rng.random() < 0.5:
        return {"messages": [_user(_pick(rng, [f"Dịch sang tiếng Anh: {vi}", f"Translate to English: {vi}"])),
                             _assistant(en)]}
    return {"messages": [_user(_pick(rng, [f"Dịch sang tiếng Việt: {en}", f"Dịch câu này: {en}"])),
                         _assistant(vi)]}


def make_idiom(rng: random.Random) -> Record:
    idiom, meaning = _pick(rng, IDIOMS)
    q = _pick(rng, [f"Thành ngữ \"{idiom}\" nghĩa là gì?", f"Giải thích \"{idiom}\"",
                    f"\"{idiom.capitalize()}\" có ý nghĩa gì?", f"Tục ngữ \"{idiom}\" dùng khi nào?"])
    return {"messages": [_user(q), _assistant(f"\"{idiom}\" {meaning}")]}


def make_word_problem(rng: random.Random) -> Record:
    kind = rng.randrange(3)
    if kind == 0:
        price, qty = _num(rng, 5, 90) * 1000, _num(rng, 2, 9)
        total = price * qty
        q = _pick(rng, [f"Một quyển vở giá {price:,} đồng. Mua {qty} quyển hết bao nhiêu tiền?".replace(",", "."),
                        f"Mỗi hộp có giá {price:,} đồng, mua {qty} hộp thì tổng bao nhiêu?".replace(",", ".")])
        a = (f"Giá một hộp là {price:,} đồng, mua {qty} hộp:\n\n"
             f"{price:,} × {qty} = **{total:,}** đồng.").replace(",", ".")
    elif kind == 1:
        a1, b1 = _num(rng, 20, 200), _num(rng, 5, 19)
        res = a1 - b1
        q = f"Một cửa hàng nhập {a1} kg gạo, buổi sáng bán {b1} kg. Còn lại bao nhiêu kg?"
        a = f"Lấy số nhập trừ số đã bán:\n\n{a1} − {b1} = **{res}** kg."
    else:
        speed, hours = _num(rng, 20, 90), _num(rng, 2, 6)
        dist = speed * hours
        q = f"Một xe chạy đều {speed} km/h trong {hours} giờ. Xe đi được bao nhiêu km?"
        a = f"Quãng đường bằng vận tốc nhân thời gian:\n\n{speed} × {hours} = **{dist}** km."
    return {"messages": [_user(q), _assistant(a)]}


# --------------------------------------------------------------------------- code: dạng bài thực tế

def make_debug(rng: random.Random) -> Record:
    kind = rng.randrange(3)
    if kind == 0:
        n = _num(rng, 5, 50)
        buggy = (f"def tong_chan(n):\n    tong = 0\n    for i in range(n):\n        if i % 2 == 0:\n            tong += i\n    return tong\n\n"
                 f"print(tong_chan({n}))")
        fixed = (f"def tong_chan(n):\n    tong = 0\n    for i in range(n + 1):\n        if i % 2 == 0:\n            tong += i\n    return tong\n\n"
                 f"print(tong_chan({n}))")
        expected = sum(i for i in range(n + 1) if i % 2 == 0)
        bug = "`range(n)` dừng ở n − 1 nên bỏ sót chính số n khi n chẵn"
    elif kind == 1:
        xs = [rng.randint(1, 40) for _ in range(6)]
        buggy = f"xs = {_fmt_list(xs)}\nprint(sum(xs) / (len(xs) - 1))"
        fixed = f"xs = {_fmt_list(xs)}\nprint(round(sum(xs) / len(xs), 4))"
        expected = round(sum(xs) / len(xs), 4)
        bug = "mẫu số dùng `len(xs) - 1` nên kết quả lớn hơn giá trị trung bình thật"
    else:
        a, b = _num(rng, 10, 99), _num(rng, 10, 99)
        buggy = f"def lon_nhat(a, b):\n    if a > b:\n        return b\n    return a\n\nprint(lon_nhat({a}, {b}))"
        fixed = f"def lon_nhat(a, b):\n    if a > b:\n        return a\n    return b\n\nprint(lon_nhat({a}, {b}))"
        expected = max(a, b)
        bug = "hai nhánh `return` bị hoán đổi nên hàm trả số nhỏ hơn"
    checks = [{"messages": [_user("x"), _assistant(f"```python\n{buggy}\n```\nbug")]}]
    check = subprocess.run([sys.executable, "-c", buggy], capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=10,
                           env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    assert check.returncode == 0, "đoạn code lỗi mẫu phải chạy được nhưng cho kết quả sai"
    ans = (f"Đoạn code chạy được nhưng sai kết quả: {bug}.\n\nSửa lại:\n\n"
           f"```python\n{fixed}\n```\nKết quả đúng là **{expected}**.")
    return {"messages": [_user(f"Code này sai ở đâu và sửa thế nào?\n\n```python\n{buggy}\n```"), _assistant(ans)]}


EXPLAIN_SNIPPETS = [
    ("def chia_an_toan(a, b):\n    if b == 0:\n        return None\n    return a / b\n\nprint(chia_an_toan(10, 4))",
     "Hàm `chia_an_toan` nhận hai số. Nếu mẫu bằng 0 thì trả `None` thay vì làm chương trình lỗi, ngược lại trả kết quả phép chia. Lời gọi `chia_an_toan(10, 4)` in ra 2.5."),
    ("def dem_tu(cau):\n    tu = cau.split()\n    return len(tu)\n\nprint(dem_tu('học python mỗi ngày'))",
     "`cau.split()` tách câu thành list các từ theo khoảng trắng, `len` đếm số phần tử. Với câu mẫu có 4 từ nên in ra 4."),
    ("def la_chan(n):\n    return n % 2 == 0\n\nprint(la_chan(7))",
     "`n % 2 == 0` trả `True` khi n chia hết cho 2. Với n = 7, phép dư là 1 nên biểu thức sai và in ra `False`."),
    ("ds = [3, 1, 4, 1, 5]\nds.sort(reverse=True)\nprint(ds[:2])",
     "`sort` sắp xếp tại chỗ theo chiều giảm dần vì có `reverse=True`. Sau đó `ds[:2]` lấy hai phần tử đầu, in ra `[5, 4]`."),
    ("def gop(a, b=None):\n    return a if b is None else a + b\n\nprint(gop(5))\nprint(gop(5, 3))",
     "Tham số `b` có mặc định `None`. Khi chỉ truyền `a`, hàm trả chính `a`; khi truyền đủ hai số, hàm trả tổng. Hai lời gọi in ra 5 và 8."),
    ("with open('ghi_chu.txt', 'w', encoding='utf-8') as f:\n    f.write('xin chào')\n\nprint(open('ghi_chu.txt', encoding='utf-8').read())",
     "`with open(..., 'w')` mở file để ghi và tự đóng khi ra khỏi khối. `f.write` ghi chuỗi vào file, sau đó đọc lại bằng `read()` và in ra `xin chào`."),
]


def make_explain_code(rng: random.Random) -> Record:
    snippet, explain = _pick(rng, EXPLAIN_SNIPPETS)
    q = _pick(rng, [f"Giải thích đoạn code này:\n\n```python\n{snippet}\n```",
                    f"Đoạn code sau làm gì?\n\n```python\n{snippet}\n```",
                    f"Đọc code này và giải thích từng bước:\n\n```python\n{snippet}\n```"])
    return {"messages": [_user(q), _assistant(explain)]}


TEST_TARGETS = [
    ("def cong(a, b):\n    return a + b", [(1, 2, 3), (0, 0, 0), (-1, 1, 0)],
     "def test_cong():\n    assert cong(1, 2) == 3\n    assert cong(0, 0) == 0\n    assert cong(-1, 1) == 0"),
    ("def la_chan(n):\n    return n % 2 == 0", [(2, True), (3, False), (0, True)],
     "def test_la_chan():\n    assert la_chan(2) is True\n    assert la_chan(3) is False\n    assert la_chan(0) is True"),
    ("def dao_nguoc(s):\n    return s[::-1]", [("ab", "ba"), ("", ""), ("xy z", "z yx")],
     "def test_dao_nguoc():\n    assert dao_nguoc('ab') == 'ba'\n    assert dao_nguoc('') == ''\n    assert dao_nguoc('xy z') == 'z yx'"),
    ("def tich(a, b):\n    return a * b", [(2, 3, 6), (0, 5, 0), (-2, 4, -8)],
     "def test_tich():\n    assert tich(2, 3) == 6\n    assert tich(0, 5) == 0\n    assert tich(-2, 4) == -8"),
    ("def lon_hon(a, b):\n    return a > b", [(3, 1, True), (1, 3, False), (2, 2, False)],
     "def test_lon_hon():\n    assert lon_hon(3, 1) is True\n    assert lon_hon(1, 3) is False\n    assert lon_hon(2, 2) is False"),
    ("def viet_hoa(s):\n    return s.upper()", [("abc", "ABC"), ("", ""), ("a b", "A B")],
     "def test_viet_hoa():\n    assert viet_hoa('abc') == 'ABC'\n    assert viet_hoa('') == ''\n    assert viet_hoa('a b') == 'A B'"),
]


def make_write_test(rng: random.Random) -> Record:
    src, cases, test = _pick(rng, TEST_TARGETS)
    q = _pick(rng, [f"Viết unit test cho hàm sau:\n\n```python\n{src}\n```",
                    f"Viết test cho code này:\n\n```python\n{src}\n```"])
    ans = (f"Dùng `assert` cho từng trường hợp:\n\n```python\nprint(\"PASS\")\n{test}\n```\n"
           "Chạy bằng `pytest` hoặc gọi thẳng `test_...()`; cả ba trường hợp đều phải qua.")
    return {"messages": [_user(q), _assistant(ans)]}


def make_refactor(rng: random.Random) -> Record:
    xs = [rng.randint(1, 50) for _ in range(6)]
    old = (f"ket_qua = []\nfor x in {_fmt_list(xs)}:\n    if x % 2 == 0:\n        ket_qua.append(x * x)\nprint(ket_qua)")
    new = f"print([x * x for x in {_fmt_list(xs)} if x % 2 == 0])"
    q = _pick(rng, [f"Rút gọn đoạn code sau:\n\n```python\n{old}\n```",
                    f"Viết lại đoạn này ngắn hơn nhưng giữ nguyên kết quả:\n\n```python\n{old}\n```"])
    ans = (f"Dùng list comprehension:\n\n```python\n{new}\n```\n"
           "Kết quả giữ nguyên, bỏ được biến tạm và vòng lặp thủ công.")
    return {"messages": [_user(q), _assistant(ans)]}


def make_tool_chain(rng: random.Random) -> Record:
    sec = _pick(rng, ["src", "docs", "configs", ".", "workspace"])
    f = _pick(rng, FILES)
    path = f if sec in (".", "workspace") else f"{sec}/{f}"
    words = _pick(rng, ["xin chào thế giới", "dữ liệu mẫu để kiểm tra", "ghi chú cuộc họp thứ ba"])
    n = len(words.split())
    out_name = _pick(rng, ["tom_tat.txt", "ket_qua.txt", "ghi_chu_tom_tat.txt"])
    return tool_record(f"Xem thư mục {sec}, tìm file {f}, đọc nó rồi lưu số từ vào {out_name}",
                       [("Bước 1: xem thư mục.", ("list_dir", {"path": sec}, "\n".join(sorted(rng.sample(FILES, 3))))),
                        ("Bước 2: mở file cần đọc.", ("read_file", {"path": path}, words)),
                        ("Bước 3: đếm số từ.", ("run_python", {"code": f"print(len('{words}'.split()))"}, str(n))),
                        ("Bước 4: ghi kết quả.", ("write_file", {"path": out_name, "content": f"Số từ: {n}"},
                                                  f"Đã ghi 9 ký tự vào {out_name}"))],
                       f"`{path}` có **{n} từ**, đã lưu vào `{out_name}`.")


def make_tool_recover(rng: random.Random) -> Record:
    good = _pick(rng, FILES)
    wrong = _pick(rng, [good.replace(".", "_cu."), f"cu/{good}", good.replace(".py", ".py.bak")])
    content = _pick(rng, ["phiên bản 2.1", "danh sách 5 việc cần làm", "cổng 9090", "chưa cấu hình"])
    q = _pick(rng, [f"Đọc file {wrong} giúp tôi", f"Cho tôi xem nội dung {wrong}"])
    return tool_record(q,
                       [("Tôi sẽ mở file đó.", ("read_file", {"path": wrong},
                                                f"LỖI: Không tìm thấy file: {wrong}")),
                        ("Không thấy file đó. Tôi đoán tên đúng là " + good + ", thử lại.",
                         ("read_file", {"path": good}, content))],
                       f"Tên `{wrong}` không tồn tại. `{good}` mới là file đúng, nội dung: {content}.")


def make_tool_no_call_hard(rng: random.Random) -> Record:
    q, ans = _pick(rng, [
        ("Làm sao để đọc một file trong Python?",
         "Dùng `open` kèm `with` để file tự đóng:\n\n```python\nwith open('data.txt', encoding='utf-8') as f:\n    noi_dung = f.read()\n```\nCần đọc theo dòng thì lặp qua `f`; cần ghi thì mở với `'w'`."),
        ("Cách ghi file trong Python là gì?",
         "Mở file ở chế độ ghi rồi gọi `write`:\n\n```python\nwith open('out.txt', 'w', encoding='utf-8') as f:\n    f.write('xin chào')\n```\n`'w'` sẽ xoá nội dung cũ; muốn ghi thêm thì dùng `'a'`."),
        ("Liệt kê file trong Python bằng code thế nào?",
         "Dùng `os.listdir` hoặc `pathlib`:\n\n```python\nfrom pathlib import Path\nfor p in sorted(Path('.').iterdir()):\n    print(p.name)\n```\n`iterdir` duyệt mọi mục trong thư mục hiện tại."),
        ("Muốn chạy code Python từ một chuỗi thì làm sao?",
         "Dùng `exec` cho câu lệnh, `eval` cho biểu thức:\n\n```python\ncode = 'x = 2 + 3'\nexec(code, ns := {})\nprint(ns['x'])\n```\nChỉ nên dùng với chuỗi do chính bạn kiểm soát."),
    ])
    q = _pick(rng, [q, f"Cho tôi hỏi: {q}", f"Bạn chỉ giúp: {q[0].lower()}{q[1:]}"])
    return {"messages": [_system_tool(), _user(q), _assistant(ans)]}


# --------------------------------------------------------------------------- code: mở rộng

def make_trace_output(rng: random.Random) -> Record:
    xs = [rng.randint(1, 30) for _ in range(5)]
    kind = rng.randrange(4)
    if kind == 0:
        code = f"xs = {_fmt_list(xs)}\nprint(xs[1:4])"
        out = repr(xs[1:4])
    elif kind == 1:
        code = f"xs = {_fmt_list(xs)}\nprint(sum(xs) - max(xs))"
        out = str(sum(xs) - max(xs))
    elif kind == 2:
        code = f"xs = {_fmt_list(xs)}\nprint(len(set(xs)))"
        out = str(len(set(xs)))
    else:
        code = f"xs = {_fmt_list(xs)}\nprint(sorted(xs)[0], sorted(xs)[-1])"
        out = f"{min(xs)} {max(xs)}"
    q = _pick(rng, [f"Đoạn code này in ra gì?\n\n```python\n{code}\n```",
                    f"Không chạy máy, hãy đoán kết quả:\n\n```python\n{code}\n```"])
    return {"messages": [_user(q), _assistant(f"Đầu ra là `{out}`.")]}


def make_string_ops(rng: random.Random) -> Record:
    s0 = _pick(rng, ["  xin chao  ", "hoc python moi ngay", "du lieu thu nghiem", "  bao cao tuan  "])
    kind = rng.randrange(4)
    tasks = ["bỏ khoảng trắng thừa", "nối các từ bằng dấu gạch", "bỏ khoảng trắng rồi viết hoa", "đếm số từ"]
    if kind == 0:
        code = f"s = {s0!r}\nprint(s.strip())"
        explain = f"`strip()` bỏ khoảng trắng hai đầu, còn `{s0.strip()}`."
    elif kind == 1:
        code = f"s = {s0!r}\nprint('-'.join(s.split()))"
        explain = f"`split()` tách theo khoảng trắng rồi `join` nối bằng dấu gạch, ra `{'-'.join(s0.split())}`."
    elif kind == 2:
        code = f"s = {s0!r}\nprint(s.strip().upper())"
        explain = f"Gộp `strip()` và `upper()`, kết quả `{s0.strip().upper()}`."
    else:
        code = f"s = {s0!r}\nprint(len(s.split()))"
        explain = f"Câu có {len(s0.split())} từ sau khi tách bằng `split()`."
    q = _pick(rng, [f"Viết code xử lý chuỗi {s0!r}: {tasks[kind]}",
                    f"Xử lý chuỗi {s0!r} giúp tôi ({tasks[kind]})"])
    return code_record(q, code, explain, rng)


def make_list_ops(rng: random.Random) -> Record:
    xs = [rng.randint(1, 9) for _ in range(8)]
    kind = rng.randrange(3)
    if kind == 0:
        ans = list(dict.fromkeys(xs))
        code = f"xs = {_fmt_list(xs)}\nprint(list(dict.fromkeys(xs)))"
        explain = f"`dict.fromkeys` giữ thứ tự xuất hiện đầu tiên và loại trùng: {_fmt_list(ans)}."
        task = "loại phần tử trùng nhưng giữ thứ tự"
    elif kind == 1:
        ans = [xs[i:i + 3] for i in range(0, len(xs), 3)]
        code = f"xs = {_fmt_list(xs)}\nprint([xs[i:i + 3] for i in range(0, len(xs), 3)])"
        explain = f"Cắt list thành các nhóm 3 phần tử: {ans}."
        task = "chia list thành các nhóm 3 phần tử"
    else:
        ans = sum(x for x in xs if x > 4)
        code = f"xs = {_fmt_list(xs)}\nprint(sum(x for x in xs if x > 4))"
        explain = f"Cộng các phần tử lớn hơn 4, được **{ans}**."
        task = "tính tổng các phần tử lớn hơn 4"
    return code_record(f"Viết code Python để {task} với danh sách {_fmt_list(xs)}", code, explain, rng)


def make_dict_ops(rng: random.Random) -> Record:
    keys = ["a", "b", "c"]
    vals = [rng.randint(1, 20) for _ in keys]
    d = dict(zip(keys, vals))
    kind = rng.randrange(3)
    if kind == 0:
        code = f"d = {d!r}\nprint(d.get('z', 0))"
        explain = "`get` với giá trị mặc định trả 0 khi key không có, không ném `KeyError`."
        task = "lấy giá trị của key 'z', nếu không có thì trả 0"
    elif kind == 1:
        code = f"d = {d!r}\nprint(sum(d.values()))"
        explain = f"`d.values()` lấy mọi giá trị, tổng là **{sum(vals)}**."
        task = "tính tổng mọi giá trị trong dict"
    else:
        code = f"d = {d!r}\nprint(max(d, key=d.get))"
        best = max(d, key=d.get)
        explain = f"`max` với `key=d.get` trả key có giá trị lớn nhất: `{best}`."
        task = "tìm key có giá trị lớn nhất"
    return code_record(f"Cho dict {d!r}, viết code để {task}", code, explain, rng)


def make_date_ops(rng: random.Random) -> Record:
    y, m, d = 2026, rng.randint(1, 12), rng.randint(1, 28)
    kind = rng.randrange(3)
    if kind == 0:
        code = f"from datetime import date\nd = date({y}, {m}, {d})\nprint(d.isoformat())"
        explain = f"`isoformat()` in ngày dạng YYYY-MM-DD: {y}-{m:02d}-{d:02d}."
        task = f"in ngày {d}/{m}/{y} theo chuẩn ISO"
    elif kind == 1:
        code = (f"from datetime import date, timedelta\nd = date({y}, {m}, {d})\n"
                "print((d + timedelta(days=7)).isoformat())")
        explain = "`timedelta(days=7)` cộng thêm một tuần rồi in lại theo ISO."
        task = f"tính ngày sau 7 ngày kể từ {d}/{m}/{y}"
    else:
        code = (f"from datetime import date\na = date({y}, {m}, {d})\nb = date({y}, 12, 28)\n"
                "print((b - a).days)")
        explain = "Trừ hai `date` cho `timedelta`, lấy `.days` ra số ngày chênh lệch."
        task = f"tính số ngày từ {d}/{m}/{y} đến 28/12/{y}"
    return code_record(f"Viết code Python {task}", code, explain, rng)


def make_regex(rng: random.Random) -> Record:
    text = _pick(rng, ["lien he: an@mail.com va binh@mail.vn", "goi 0912345678 hoac 0987654321",
                       "ma don: A123, B456, C789"])
    if rng.random() < 0.5:
        code = f"import re\ntext = {text!r}\nprint(re.findall(r'[\\w.]+@[\\w.]+', text))"
        explain = "Mẫu `[\\w.]+@[\\w.]+` bắt các chuỗi có dạng email trong văn bản."
        task = "trích mọi email"
    else:
        code = f"import re\ntext = {text!r}\nprint(re.findall(r'\\d+', text))"
        explain = "`\\d+` bắt từng dãy chữ số liên tiếp."
        task = "trích mọi dãy số"
    return code_record(f"Dùng regex để {task} trong chuỗi: {text}", code, explain, rng)


def make_algo(rng: random.Random) -> Record:
    import math
    kind = rng.randrange(3)
    if kind == 0:
        xs = sorted(rng.sample(range(1, 100), 9))
        target = rng.choice(xs)
        code = ("def tim_nhi_phan(xs, x):\n    lo, hi = 0, len(xs) - 1\n"
                "    while lo <= hi:\n        mid = (lo + hi) // 2\n"
                "        if xs[mid] == x:\n            return mid\n"
                "        if xs[mid] < x:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n"
                "    return -1\n\n"
                f"print(tim_nhi_phan({_fmt_list(xs)}, {target}))")
        explain = (f"Tìm nhị phân chia đôi khoảng tìm mỗi bước, độ phức tạp O(log n). "
                   f"Vị trí của {target} là **{xs.index(target)}**.")
        task = f"viết tìm nhị phân và tìm {target} trong danh sách đã sắp xếp {_fmt_list(xs)}"
    elif kind == 1:
        a, b = _num(rng, 10, 500), _num(rng, 10, 500)
        code = (f"def ucln(a, b):\n    while b:\n        a, b = b, a % b\n    return a\n\n"
                f"print(ucln({a}, {b}))")
        explain = f"Thuật toán Euclid lặp phép lấy dư tới khi b bằng 0. UCLN({a}, {b}) = **{math.gcd(a, b)}**."
        task = f"viết thuật toán Euclid tính UCLN của {a} và {b}"
    else:
        s0 = _pick(rng, ["madam", "level", "python", "radar", "abcba"])
        code = f"def la_palindrome(s):\n    return s == s[::-1]\n\nprint(la_palindrome({s0!r}))"
        explain = f"So chuỗi với bản đảo ngược; `{s0}` {'là' if s0 == s0[::-1] else 'không phải'} palindrome."
        task = f"kiểm tra chuỗi '{s0}' có phải palindrome"
    return code_record(f"Viết code Python: {task}", code, explain, rng)


def make_error_explain(rng: random.Random) -> Record:
    kind = rng.randrange(3)
    if kind == 0:
        bad = "xs = [1, 2, 3]\nprint(xs[5])"
        err = "IndexError: list index out of range"
        fixed = "xs = [1, 2, 3]\nprint(xs[-1])"
        why = "list chỉ có 3 phần tử nên chỉ số hợp lệ là 0..2; `xs[5]` vượt biên"
    elif kind == 1:
        bad = "d = {'a': 1}\nprint(d['b'])"
        err = "KeyError: 'b'"
        fixed = "d = {'a': 1}\nprint(d.get('b', 0))"
        why = "key 'b' không có trong dict"
    else:
        bad = "print('tong: ' + 5)"
        err = "TypeError: can only concatenate str (not \"int\") to str"
        fixed = "print('tong: ' + str(5))"
        why = "không cộng trực tiếp chuỗi với số nguyên"
    q = _pick(rng, [f"Code này báo `{err}`. Vì sao và sửa thế nào?\n\n```python\n{bad}\n```",
                    f"Gặp lỗi `{err}` với đoạn sau, xử lý ra sao?\n\n```python\n{bad}\n```"])
    ans = f"Nguyên nhân: {why}.\n\nSửa lại:\n\n```python\n{fixed}\n```\nĐoạn trên chạy được và không còn lỗi."
    return {"messages": [_user(q), _assistant(ans)]}


def make_complexity(rng: random.Random) -> Record:
    n = _num(rng, 10, 200)
    kind = rng.randrange(3)
    if kind == 0:
        code = f"xs = list(range({n}))\ntong = 0\nfor x in xs:\n    tong += x\nprint(tong)"
        big, why = "O(n)", "một vòng lặp duyệt đúng n phần tử"
    elif kind == 1:
        code = (f"xs = list(range({n}))\ndem = 0\nfor i in xs:\n    for j in xs:\n"
                "        dem += 1\nprint(dem)")
        big, why = "O(n²)", "hai vòng lặp lồng nhau, mỗi cặp (i, j) được xét một lần"
    else:
        code = f"xs = list(range({n}))\nprint(xs[{n // 2}])"
        big, why = "O(1)", "truy cập theo chỉ số của list là hằng số, không phụ thuộc n"
    q = _pick(rng, [f"Độ phức tạp của đoạn code này là gì?\n\n```python\n{code}\n```",
                    f"Phân tích Big-O:\n\n```python\n{code}\n```"])
    return {"messages": [_user(q), _assistant(f"Độ phức tạp thời gian là **{big}** vì {why}.")]}


# --------------------------------------------------------------------------- suy luận từng bước

def make_reasoning(rng: random.Random) -> Record:
    kind = rng.randrange(4)
    if kind == 0:
        boxes, per, sold = _num(rng, 3, 20), _num(rng, 5, 30), _num(rng, 1, 40)
        total = boxes * per
        left = total - sold
        q = f"Một cửa hàng có {boxes} thùng, mỗi thùng {per} chai. Bán được {sold} chai. Còn lại bao nhiêu chai?"
        steps = [f"Tổng số chai: {boxes} × {per} = {total}",
                 f"Số chai còn lại: {total} − {sold} = {left}"]
        final, unit = left, "chai"
    elif kind == 1:
        price, disc = _num(rng, 50, 400) * 1000, _pick(rng, [10, 15, 20, 25, 30])
        cut = price * disc // 100
        pay = price - cut
        q = f"Một món giá {price:,} đồng, giảm {disc}%. Phải trả bao nhiêu?".replace(",", ".")
        steps = [f"Tiền giảm: {price:,} × {disc}% = {cut:,}".replace(",", "."),
                 f"Số phải trả: {price:,} − {cut:,} = {pay:,}".replace(",", ".")]
        final, unit = pay, "đồng"
    elif kind == 2:
        speed, h1, h2 = _num(rng, 30, 80), _num(rng, 1, 4), _num(rng, 1, 4)
        d1, d2 = speed * h1, speed * h2
        q = f"Xe chạy {speed} km/h, đi {h1} giờ buổi sáng và {h2} giờ buổi chiều. Tổng quãng đường?"
        steps = [f"Buổi sáng: {speed} × {h1} = {d1} km",
                 f"Buổi chiều: {speed} × {h2} = {d2} km",
                 f"Tổng: {d1} + {d2} = {d1 + d2} km"]
        final, unit = d1 + d2, "km"
    else:
        workers, days = _num(rng, 2, 12), _num(rng, 2, 15)
        work = workers * days
        new_workers = _pick(rng, [w for w in range(2, 13) if work % w == 0] or [workers])
        new_days = work // new_workers
        q = (f"{workers} người làm xong một việc trong {days} ngày. "
             f"Nếu có {new_workers} người cùng năng suất thì mất bao nhiêu ngày?")
        steps = [f"Tổng khối lượng: {workers} × {days} = {work} người-ngày",
                 f"Với {new_workers} người: {work} ÷ {new_workers} = {new_days} ngày"]
        final, unit = new_days, "ngày"
    body = "\n".join(f"Bước {i}: {t}" for i, t in enumerate(steps, 1))
    return {"messages": [_user(q), _assistant(f"{body}\n\nKết quả: **{final}** {unit}.")]}


def make_reasoning_code(rng: random.Random) -> Record:
    a, b, c = _num(rng, 2, 30), _num(rng, 2, 30), _num(rng, 2, 12)
    total = a * b
    left = total - c
    code = (f"so_thung = {a}\nchai_moi_thung = {b}\nda_ban = {c}\n"
            f"tong = so_thung * chai_moi_thung\ncon_lai = tong - da_ban\nprint(con_lai)")
    q = (f"Có {a} thùng, mỗi thùng {b} chai, đã bán {c} chai. "
         "Viết code Python tính số chai còn lại, giải thích từng bước.")
    ans = (f"Chia bài toán thành hai bước: tính tổng rồi trừ phần đã bán.\n\n"
           f"```python\n{code}\n```\n"
           f"Bước 1: tổng = {a} × {b} = {total}. Bước 2: còn lại = {total} − {c} = **{left}**.")
    return {"messages": [_user(q), _assistant(ans)]}


CHAT_TEMPLATES = [
    Template("reasoning", 60, make_reasoning),
    Template("multiturn", 45, make_multiturn),
    Template("format_follow", 20, make_format_follow),
    Template("clarify", 12, make_clarify),
    Template("unknown", 10, make_unknown),
    Template("rewrite", 12, make_rewrite),
    Template("translate", 24, make_translate),
    Template("idiom", 22, make_idiom),
    Template("word_problem", 35, make_word_problem),
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
    Template("trace_output", 45, make_trace_output),
    Template("string_ops", 40, make_string_ops),
    Template("list_ops", 40, make_list_ops),
    Template("dict_ops", 35, make_dict_ops),
    Template("date_ops", 30, make_date_ops),
    Template("regex", 25, make_regex),
    Template("algo", 40, make_algo),
    Template("error_explain", 30, make_error_explain),
    Template("complexity", 25, make_complexity),
    Template("reasoning_code", 35, make_reasoning_code),
    Template("debug", 40, make_debug),
    Template("explain_code", 14, make_explain_code),
    Template("write_test", 10, make_write_test),
    Template("refactor", 25, make_refactor),
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
    Template("chain", 30, make_tool_chain),
    Template("recover", 20, make_tool_recover),
    Template("no_call_hard", 12, make_tool_no_call_hard),
    Template("read", 40, make_tool_read),
    Template("write", 30, make_tool_write),
    Template("list", 30, make_tool_list),
    Template("run_python", 30, make_tool_run_python),
    Template("multi", 25, make_tool_multi),
    Template("error", 15, make_tool_error),
    Template("search_files", 10, make_tool_search_files),
    Template("search_code", 10, make_tool_search_code),
    Template("project_tree", 8, make_tool_project_tree),
    Template("preview", 8, make_tool_preview),
    Template("no_call", 20, make_tool_no_call),
]


def _question_key(rec: Record) -> str:
    """Khoá khử trùng theo toàn bộ lượt người dùng (mẫu nhiều lượt cần phân biệt cả lượt sau)."""
    turns = [m["content"].strip().lower() for m in rec["messages"] if m["role"] == "user"]
    return " || ".join(turns)


def raw_question_keys(raw_dir: Path) -> set[str]:
    keys = set()
    for path in sorted(raw_dir.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                keys.add(_question_key(json.loads(line)))
    return keys


def generate(templates: list[Template], seed: int, skip: set[str] | None = None,
             max_attempts_per_sample: int = 50, seen: set[str] | None = None) -> list[Record]:
    rng = random.Random(seed)
    skip = skip if skip is not None else set()
    seen = seen if seen is not None else set()
    out: list[Record] = []
    for tpl in templates:
        made = 0
        attempts = 0
        while made < tpl.count:
            if attempts > tpl.count * max_attempts_per_sample:
                print(f"  [canh bao] khuon '{tpl.name}' het bien the: sinh {made}/{tpl.count} mau")
                break
            attempts += 1
            rec = tpl.make(rng)
            key = json.dumps(rec, ensure_ascii=False, sort_keys=True)
            qkey = _question_key(rec)
            if key in tpl.seen or qkey in skip or qkey in seen:
                continue
            tpl.seen.add(key)
            seen.add(qkey)
            out.append(rec)
            made += 1
    rng.shuffle(out)
    return out


CODE_FENCE = re.compile(r"```python\n(.*?)\n```", re.DOTALL)


@lru_cache(maxsize=2048)
def _verify_code_cached(code: str, answer: str, timeout_s: int) -> tuple[str, ...]:
    """Cache kiểm chứng theo nội dung để không chạy lại cùng mẫu ở các lần build giống seed."""
    with tempfile.TemporaryDirectory() as workdir:
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-c", code],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=timeout_s, cwd=workdir, stdin=subprocess.DEVNULL,
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            )
        except subprocess.TimeoutExpired:
            return ("quá thời gian",)
    stdout = proc.stdout
    if proc.returncode != 0:
        last = proc.stderr.strip().splitlines()[-1][:120] if proc.stderr.strip() else "mã thoát khác 0"
        return (f"lỗi khi chạy: {last}",)
    errors = []
    claimed = re.search(r"Đầu ra là `([^`]+)`", answer)
    if claimed and claimed.group(1).strip() != stdout.strip():
        errors.append(f"đầu ra nêu là {claimed.group(1)!r} nhưng chạy ra {stdout.strip()!r}")
    numbers = re.findall(r"\*\*(-?[\d.,]+)\*\*", answer)
    errors += [f"kết quả {n} không có trong đầu ra" for n in numbers
               if n not in stdout and n.replace(",", "") not in stdout]
    return tuple(errors)


def verify_code(record: Record, timeout_s: int = 5) -> list[str]:
    """Chạy code trong câu trả lời (hoặc trong câu hỏi nếu đáp án chỉ nêu kết quả) và đối chiếu."""
    answer = record["messages"][-1]["content"]
    blocks = CODE_FENCE.findall(answer)
    if not blocks:
        for m in record["messages"][:-1]:
            blocks = CODE_FENCE.findall(m["content"])
            if blocks:
                break
    if not blocks:
        return ["không có khối ```python```"]
    return list(_verify_code_cached(blocks[0], answer, int(timeout_s)))


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


def quality_report(data: dict[str, list[Record]]) -> None:
    """In thống kê chất lượng để so sánh giữa các lần sinh."""
    print(f"{'file':<12} {'mẫu':>6} {'câu hỏi riêng':>14} {'câu trả lời riêng':>19} {'lượt/TB':>8} {'từ/TB':>7}")
    for name, rows in data.items():
        qkeys = {_question_key(r) for r in rows}
        answers = {r["messages"][-1]["content"] for r in rows}
        turns = sum(sum(1 for m in r["messages"] if m["role"] == "assistant") for r in rows) / len(rows)
        words = sum(len(r["messages"][-1]["content"].split()) for r in rows) / len(rows)
        print(f"{name:<12} {len(rows):>6} {len(qkeys):>14} {len(answers):>19} {turns:>8.2f} {words:>7.1f}")


def build(seed: int = 2026, scale: float = 1.0, raw_dir: str | Path = "data/raw") -> dict[str, list[Record]]:
    def scaled(templates):
        return [Template(t.name, max(1, int(t.count * scale)), t.make) for t in templates]

    skip = raw_question_keys(Path(raw_dir))
    seen: set[str] = set()
    data = {
        "tool_call": generate(scaled(TOOL_TEMPLATES), seed + 2, skip, seen=seen),
        "code_vi": generate(scaled(CODE_TEMPLATES), seed + 1, skip, seen=seen),
        "chat_vi": generate(scaled(CHAT_TEMPLATES), seed, skip, seen=seen),
    }
    failed = verify_all(data)
    if failed:
        raise RuntimeError(f"{failed} mẫu code chạy sai; sửa khuôn trước khi sinh tiếp")
    return data


def mix_for_adapters(data: dict[str, list[Record]], seed: int = 2026, chat_share: float = 0.35):
    """Bản trộn cho từng LoRA: phần chính của adapter + một phần chat để giữ tiếng Việt chung."""
    rng = random.Random(seed + 9)
    chat = list(data["chat_vi"])
    rng.shuffle(chat)
    keep = chat[: int(len(chat) * chat_share)]
    return {
        "train_agent": data["tool_call"] + keep,
        "train_code": data["code_vi"] + keep,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data/generated")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--scale", type=float, default=1.0, help="nhân số mẫu, ví dụ 1.4")
    parser.add_argument("--chat-share", type=float, default=0.35,
                        help="tỉ lệ chat trộn vào bản train của từng adapter")
    args = parser.parse_args()

    data = build(args.seed, args.scale)
    mixed = mix_for_adapters(data, args.seed, args.chat_share)
    out = Path(args.out_dir)
    for name, rows in {**data, **mixed}.items():
        write_jsonl(out / f"{name}.jsonl", rows)
    quality_report(data)
    print(" | ".join(f"{k}: {len(v)}" for k, v in mixed.items()) + " (file trộn cho từng LoRA)")
    total = sum(len(r) for r in data.values())
    print(f"Tổng mẫu sinh mới: {total} | đã ghi vào {out}")


if __name__ == "__main__":
    main()
