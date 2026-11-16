"""Lập kế hoạch nhẹ theo quy tắc; không nạp thêm model trên máy 8GB."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PlanStep:
    number: int
    title: str
    description: str


def make_plan(task: str) -> list[PlanStep]:
    text = task.lower()
    steps: list[tuple[str, str]] = []
    if any(x in text for x in ("project", "dự án", "toàn bộ code", "nhiều file")):
        steps.append(("Khảo sát", "Xác định cấu trúc và các file liên quan."))
    if any(x in text for x in ("sửa", "bug", "lỗi", "debug")):
        steps.extend([
            ("Phân tích", "Đọc luồng xử lý và xác định nguyên nhân có bằng chứng."),
            ("Đề xuất thay đổi", "Tạo thay đổi nhỏ, giữ tương thích hiện có."),
        ])
    if any(x in text for x in ("viết", "tạo", "xây dựng", "thêm", "update", "nâng cấp")):
        steps.append(("Triển khai", "Thực hiện theo từng phần và kiểm tra đầu vào."))
    if any(x in text for x in ("test", "kiểm thử", "chạy thử", "sửa")) or steps:
        steps.append(("Kiểm chứng", "Chạy kiểm thử phù hợp và báo cáo giới hạn."))
    if not steps:
        steps = [("Hiểu yêu cầu", "Làm rõ mục tiêu và ràng buộc."),
                 ("Trả lời", "Đưa ra kết quả hoặc bước tiếp theo có căn cứ.")]
    # Bỏ tiêu đề trùng, tối đa 5 bước.
    unique: list[tuple[str, str]] = []
    for item in steps:
        if item[0] not in [x[0] for x in unique]:
            unique.append(item)
    return [PlanStep(i + 1, title, desc) for i, (title, desc) in enumerate(unique[:5])]
