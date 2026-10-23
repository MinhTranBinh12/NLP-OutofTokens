"""
Module tiền xử lý văn bản tiếng Việt cho bài toán ViCloABSA.

Hỗ trợ 2 chế độ:
1. 'minimal' (Mặc định - Baseline lần 1):
   - Chuẩn hóa Unicode NFC.
   - Tách dấu câu dính chữ (':', ',').
   - Giữ nguyên Teen Code, từ viết tắt, emoji.

2. 'normalized' (Pipeline tiền xử lý nâng cao):
   - Chuẩn hóa Unicode NFC.
   - Bóc tách bẫy số tiền 'k' (\d+k -> \d+ nghìn) trước khi thay thế teencode.
   - Rút gọn ký tự lặp kéo dài cảm xúc ('đẹpppp' -> 'đẹp', 'quáaaa' -> 'quá').
   - Chuẩn hóa Teencode & từ viết tắt phổ biến trong review thời trang TMĐT.
   - Ánh xạ Emoji sang từ biểu cảm tiếng Việt (giúp PhoBERT không bị biến emoji thành <unk>).
   - Tách dấu câu dính chữ và làm sạch khoảng trắng thừa.
"""

import re
import unicodedata
from typing import Dict

# 1. Từ điển Teencode & viết tắt thương mại điện tử
TEENCODE_DICT: Dict[str, str] = {
    "sp": "sản phẩm",
    "s/p": "sản phẩm",
    "sz": "size",
    "đc": "được",
    "dc": "được",
    "k": "không",
    "ko": "không",
    "kh": "không",
    "khg": "không",
    "hok": "không",
    "hong": "không",
    "hông": "không",
    "hơm": "không",
    "k0": "không",
    "mik": "mình",
    "mk": "mình",
    "tui": "tôi",
    "sop": "shop",
    "shopp": "shop",
    "shopee": "shopee",
    "ship": "giao hàng",
    "shipper": "người giao hàng",
    "shiper": "người giao hàng",
    "nv": "nhân viên",
    "nvien": "nhân viên",
    "cskh": "chăm sóc khách hàng",
    "ib": "nhắn tin",
    "inbox": "nhắn tin",
    "rep": "phản hồi",
    "tl": "trả lời",
    "tloi": "trả lời",
    "cl": "chất lượng",
    "bth": "bình thường",
    "bt": "bình thường",
    "nhìu": "nhiều",
    "thui": "thôi",
    "thoii": "thôi",
    "rùi": "rồi",
    "roii": "rồi",
    "vs": "với",
    "cx": "cũng",
    "cug": "cũng",
    "z": "vậy",
    "zậy": "vậy",
    "mn": "mọi người",
    "mng": "mọi người",
    "tks": "cảm ơn",
    "thks": "cảm ơn",
    "cmon": "cảm ơn",
    "hịn": "xịn",
    "xink": "xinh",
    "vcl": "rất",
    "vl": "rất",
    "vđ": "rất",
    "vkl": "rất",
    "đth": "điện thoại",
    "dt": "điện thoại",
    "tks shop": "cảm ơn shop",
    "sẽ ủng hộ": "sẽ ủng hộ",
}

# 2. Bảng ánh xạ Emoji phổ biến sang từ mang sắc thái cảm xúc tiếng Việt
EMOJI_DICT: Dict[str, str] = {
    "❤️": " yêu thích ",
    "💖": " yêu thích ",
    "💕": " yêu thích ",
    "💓": " yêu thích ",
    "😍": " yêu thích mê ",
    "🥰": " yêu thích ",
    "😘": " yêu thích ",
    "👍": " tốt hài lòng ",
    "👌": " tốt ổn ",
    "👏": " khen ngợi ",
    "💯": " tuyệt đối chất lượng ",
    "🔥": " xuất sắc ",
    "✨": " đẹp lung linh ",
    "⭐": " sao ",
    "🌟": " sao ",
    "😊": " vui vẻ ",
    "😁": " vui vẻ ",
    "😃": " vui vẻ ",
    "😆": " vui vẻ ",
    "😭": " thất vọng buồn ",
    "😢": " buồn ",
    "🥺": " tiếc nuối ",
    "😞": " thất vọng ",
    "😔": " thất vọng ",
    "😡": " rất tệ tức giận ",
    "🤬": " rất tệ ",
    "👿": " rất tệ ",
    "👎": " tệ không hài lòng ",
    "😤": " bực mình ",
    "😠": " tức giận ",
}


def normalize_nfc(text: str) -> str:
    """Chuẩn hóa văn bản về Unicode dạng NFC chuẩn."""
    if not isinstance(text, str):
        return ""
    return unicodedata.normalize("NFC", text)


def separate_stuck_punctuation(text: str) -> str:
    """Tách dấu câu (':', ',', '.', '!') nếu dính liền với chữ."""
    text = re.sub(r'(:)(?=[^\s])', r' : ', text)
    text = re.sub(r'([^\s])(:)', r'\1 : ', text)
    text = re.sub(r'(,)(?=[^\s])', r' , ', text)
    text = re.sub(r'(!+)', r' ! ', text)
    text = re.sub(r'(\?+)', r' ? ', text)
    return text


def resolve_price_trap(text: str) -> str:
    """
    Xử lý bẫy chữ 'k' giá tiền: 92k, 150k, 200 k -> 92 nghìn, 150 nghìn.
    Thực hiện TRƯỚC KHI chuyển đổi teencode (tránh biến '92k' thành '92 không').
    """
    text = re.sub(r'\b(\d+)\s*[kK]\b', r'\1 nghìn', text)
    # Xử lý đánh giá sao: 5*, 5sao -> 5 sao
    text = re.sub(r'\b([1-5])\s*\*\b', r'\1 sao', text)
    return text


def remove_elongated_chars(text: str) -> str:
    """
    Rút gọn ký tự lặp kéo dài cảm xúc:
    'đẹpppp' -> 'đẹp', 'quáaaaa' -> 'quá', 'rấtttt' -> 'rất'.
    """
    return re.sub(r'([a-zA-Zà-ỹÀ-Ỹ])\1{2,}', r'\1', text)


def normalize_emojis(text: str) -> str:
    """Ánh xạ các emoji phổ biến thành từ ngữ biểu cảm tiếng Việt."""
    for emoji, sentiment_word in EMOJI_DICT.items():
        if emoji in text:
            text = text.replace(emoji, sentiment_word)
    return text


def normalize_teencode(text: str) -> str:
    """
    Chuẩn hóa Teencode theo ranh giới từ nguyên (\bword\b),
    không làm ảnh hưởng tới các từ ghép có chứa chuỗi đó.
    """
    words = text.split()
    normalized_words = []
    for w in words:
        w_lower = w.lower()
        if w_lower in TEENCODE_DICT:
            # Thay thế nhưng giữ phong cách chữ hoa nếu có
            replacement = TEENCODE_DICT[w_lower]
            normalized_words.append(replacement)
        else:
            normalized_words.append(w)
    return " ".join(normalized_words)


def clean_text_advanced(text: str) -> str:
    """
    Toàn bộ quy trình chuẩn hóa văn bản nâng cao (Pipeline 2):
    1. Unicode NFC
    2. Tách dấu câu dính chữ
    3. Xử lý bẫy chữ k giá tiền & số sao (92k -> 92 nghìn, 5* -> 5 sao)
    4. Rút gọn ký tự kéo dài (đẹpppp -> đẹp)
    5. Ánh xạ Emoji sang từ cảm xúc tiếng Việt
    6. Chuẩn hóa Teencode và từ viết tắt
    7. Làm sạch khoảng trắng thừa
    """
    if not isinstance(text, str):
        return ""
    text = normalize_nfc(text)
    text = separate_stuck_punctuation(text)
    text = resolve_price_trap(text)
    text = remove_elongated_chars(text)
    text = normalize_emojis(text)
    text = normalize_teencode(text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def preprocess_text(text: str, mode: str = "minimal") -> str:
    """
    Hàm tiền xử lý hợp nhất:
    - mode="minimal": Chỉ NFC và tách dấu câu dính (Baseline Exp 1).
    - mode="normalized": Chuẩn hóa nâng cao toàn diện (Exp 3).
    """
    if mode == "normalized":
        return clean_text_advanced(text)
    # Mặc định minimal
    text = normalize_nfc(text)
    text = separate_stuck_punctuation(text)
    return re.sub(r'\s+', ' ', text).strip()
