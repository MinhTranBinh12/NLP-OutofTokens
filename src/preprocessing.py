"""
Module tiền xử lý văn bản tiếng Việt cho bài toán ViCloABSA.
Tuân thủ nguyên tắc 'Minimal Preprocessing':
- Chuẩn hóa Unicode NFC.
- Tách các dấu câu dính chữ (như ':' và ',') để tokenizer không bị dính từ.
- Giữ nguyên Teen Code, từ viết tắt, emoji, dấu tiếng Việt để giữ trọn vẹn ngữ cảnh cảm xúc.
"""

import re
import unicodedata

def normalize_nfc(text: str) -> str:
    """
    Chuẩn hóa văn bản về Unicode dạng NFC chuẩn.
    Tránh lỗi ký tự tổ hợp (NFD) thường gặp trên bàn phím tiếng Việt.
    """
    if not isinstance(text, str):
        return ""
    return unicodedata.normalize("NFC", text)

def separate_stuck_punctuation(text: str) -> str:
    """
    Tách các dấu câu dính liền với từ (ví dụ: 'chất liệu:ổn', 'đẹp,chất lượng').
    Thêm khoảng trắng quanh dấu ':' và ',' để các token không bị dính vào dấu câu.
    """
    # Thêm khoảng trắng sau dấu hai chấm nếu theo sau là chữ/số
    text = re.sub(r'(:)(?=[^\s])', r' : ', text)
    # Thêm khoảng trắng trước dấu hai chấm nếu phía trước là chữ/số
    text = re.sub(r'([^\s])(:)', r'\1 : ', text)
    # Thêm khoảng trắng sau dấu phẩy nếu dính liền
    text = re.sub(r'(,)(?=[^\s])', r' , ', text)
    # Chuẩn hóa lại các khoảng trắng liên tiếp
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_text(text: str, separate_punct: bool = False) -> str:
    """
    Hàm tiền xử lý tổng thể cho văn bản review.
    Lưu ý: Nếu separate_punct=True, độ dài chuỗi có thể thay đổi,
    chỉ nên áp dụng trước khi gán nhãn nếu điều chỉnh lại tọa độ span,
    hoặc áp dụng độc lập cho nhánh phân loại Sentence-level.
    """
    text = normalize_nfc(text)
    if separate_punct:
        text = separate_stuck_punctuation(text)
    return text.strip()
