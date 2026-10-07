"""
Module xử lý nhãn (Label Processing) cho bài toán ViCloABSA:
1. Chuyển đổi Spans sang Vector gán nhãn 5 chiều (Sentence-level):
   - 5 Khía cạnh: Material, General, Service, Design, Price.
   - 4 Trạng thái: 0: None, 1: positive, 2: neutral, 3: negative.
   - Quy tắc trung bình/hòa hoãn khi một comment có nhiều cảm xúc trên cùng khía cạnh:
     + Nếu có cả 'positive' và 'negative' -> gộp thành 'neutral' (lấy trung bình bù trừ).
     + Nếu có '+' hoặc '-' xuất hiện chung với 'neutral' -> giữ '+' hoặc '-'.
     + Nếu toàn bộ cùng loại -> giữ nguyên loại đó.
     + Nếu khía cạnh không được nhắc đến -> 0 (None).
2. Chuyển đổi Spans sang Chuỗi nhãn BIO (Span-level / Sequence Labeling):
   - Lưu trữ tọa độ ký tự [start, end] của từng token trong văn bản gốc.
   - Không gian 31 nhãn BIO (1 nhãn 'O' + 5 aspects x 3 sentiments x 2 prefix B-/I-).
   - Hỗ trợ giải mã ngược (decoding) từ chuỗi nhãn BIO về danh sách spans.
"""

import re
from typing import List, Dict, Tuple, Any, Optional

# Danh sách 5 khía cạnh theo thứ tự cố định của đồ án
ASPECTS: List[str] = ["Material", "General", "Service", "Design", "Price"]

# Không gian 3 cảm xúc của thực thể span
SENTIMENTS: List[str] = ["positive", "neutral", "negative"]

# Mã hóa 4 trạng thái cảm xúc cấp độ câu (Sentence-Level)
SENTIMENT_TO_ID: Dict[str, int] = {
    "None": 0,
    "positive": 1,
    "neutral": 2,
    "negative": 3,
}

ID_TO_SENTIMENT: Dict[int, str] = {v: k for k, v in SENTIMENT_TO_ID.items()}

# Xây dựng danh sách 31 nhãn BIO
BIO_TAGS: List[str] = ["O"]
for asp in ASPECTS:
    for sent in SENTIMENTS:
        BIO_TAGS.append(f"B-{asp}#{sent}")
        BIO_TAGS.append(f"I-{asp}#{sent}")

BIO_TAG2ID: Dict[str, int] = {tag: idx for idx, tag in enumerate(BIO_TAGS)}
ID2BIO_TAG: Dict[int, str] = {idx: tag for idx, tag in enumerate(BIO_TAGS)}


def resolve_aspect_sentiment(sentiments: List[str]) -> str:
    """
    Quy tắc giải quyết xung đột cảm xúc cho cùng một khía cạnh trong một comment:
    - Nếu có cả 'positive' và 'negative' -> trung bình thành 'neutral'.
    - Nếu có 'positive' đi chung với 'neutral' (không có negative) -> 'positive'.
    - Nếu có 'negative' đi chung với 'neutral' (không có positive) -> 'negative'.
    - Nếu các nhãn cùng loại -> giữ nguyên loại đó.
    - Nếu rỗng -> 'None'.
    """
    if not sentiments:
        return "None"
    
    sent_set = set(sentiments)
    
    # 1. Có cả Positive và Negative -> Trung bình bù trừ thành Neutral
    if "positive" in sent_set and "negative" in sent_set:
        return "neutral"
    
    # 2. Có Positive (và có thể có Neutral, nhưng không có Negative) -> Positive
    elif "positive" in sent_set:
        return "positive"
    
    # 3. Có Negative (và có thể có Neutral, nhưng không có Positive) -> Negative
    elif "negative" in sent_set:
        return "negative"
    
    # 4. Chỉ có Neutral
    elif "neutral" in sent_set:
        return "neutral"
    
    return "None"


def spans_to_aspect_vector(spans: List[List[Any]]) -> List[int]:
    """
    Chuyển đổi danh sách spans [[start, end, 'Aspect#sentiment'], ...]
    thành vector 5 chiều: [y_Material, y_General, y_Service, y_Design, y_Price] in {0, 1, 2, 3}^5.
    
    Áp dụng quy tắc trung bình/hòa hoãn cảm xúc nếu cùng 1 aspect có nhiều spans.
    """
    aspect_sentiments: Dict[str, List[str]] = {asp: [] for asp in ASPECTS}
    
    for span in spans:
        label = span[2]
        if "#" in label:
            aspect, sentiment = label.split("#", 1)
            if aspect in aspect_sentiments:
                aspect_sentiments[aspect].append(sentiment)
    
    vector = []
    for asp in ASPECTS:
        final_sentiment = resolve_aspect_sentiment(aspect_sentiments[asp])
        vector.append(SENTIMENT_TO_ID[final_sentiment])
        
    return vector


def tokenize_with_offsets(text: str) -> List[Dict[str, Any]]:
    """
    Tách từ/âm tiết và dấu câu từ văn bản, lưu lại chính xác vị trí ký tự gốc [start, end].
    Hàm này không làm thay đổi văn bản gốc, đảm bảo offset khớp 100% với các spans trong JSON.
    """
    tokens = []
    # Khớp từ (chữ cái/chữ số/ký tự tiếng Việt) hoặc ký tự dấu câu riêng lẻ
    for match in re.finditer(r'\w+|[^\w\s]', text):
        tokens.append({
            "text": match.group(),
            "start": match.start(),
            "end": match.end(),
        })
    return tokens


def align_spans_to_bio(tokens: List[Dict[str, Any]], spans: List[List[Any]]) -> Tuple[List[str], List[int]]:
    """
    Ánh xạ các spans vị trí [start_char, end_char, 'Aspect#sentiment'] 
    vào chuỗi các nhãn BIO cho từng token.
    
    Trả về:
    - bio_tags: Danh sách nhãn chuỗi (ví dụ: ['O', 'B-Material#positive', ...])
    - bio_tag_ids: Danh sách mã số tương ứng [0..30]
    """
    bio_tags = ["O"] * len(tokens)
    
    # Sắp xếp các spans theo vị trí bắt đầu
    sorted_spans = sorted(spans, key=lambda s: (s[0], s[1]))
    
    for span in sorted_spans:
        s_start, s_end, label = span[0], span[1], span[2]
        
        # Tìm tất cả token có phần giao thoa ký tự với span
        matching_token_indices = []
        for idx, tok in enumerate(tokens):
            # Điều kiện giao thoa ký tự: max(t_start, s_start) < min(t_end, s_end)
            if max(tok["start"], s_start) < min(tok["end"], s_end):
                matching_token_indices.append(idx)
        
        # Gán nhãn B- cho token đầu tiên, I- cho các token tiếp theo
        for i, idx in enumerate(matching_token_indices):
            if i == 0:
                bio_tags[idx] = f"B-{label}"
            else:
                bio_tags[idx] = f"I-{label}"
                
    bio_tag_ids = [BIO_TAG2ID.get(tag, 0) for tag in bio_tags]
    return bio_tags, bio_tag_ids


def bio_to_spans(tokens: List[Dict[str, Any]], bio_tags: List[str]) -> List[Dict[str, Any]]:
    """
    Giải mã ngược từ chuỗi nhãn BIO và danh sách tokens về lại danh sách các spans.
    Mỗi span gồm: {'start': int, 'end': int, 'aspect': str, 'sentiment': str, 'label': str, 'text': str}
    """
    extracted_spans: List[Dict[str, Any]] = []
    curr_label: Optional[str] = None
    curr_start: Optional[int] = None
    curr_end: Optional[int] = None
    
    for tok, tag in zip(tokens, bio_tags):
        if tag.startswith("B-"):
            # Nếu đang có span trước đó chưa đóng -> đóng span
            if curr_label is not None:
                asp, sent = curr_label.split("#", 1) if "#" in curr_label else (curr_label, "neutral")
                extracted_spans.append({
                    "start": curr_start,
                    "end": curr_end,
                    "aspect": asp,
                    "sentiment": sent,
                    "label": curr_label,
                })
            curr_label = tag[2:]
            curr_start = tok["start"]
            curr_end = tok["end"]
            
        elif tag.startswith("I-"):
            label = tag[2:]
            if curr_label == label:
                # Kéo dài span hiện tại
                curr_end = tok["end"]
            else:
                # Trường hợp nhãn I- không khớp với nhãn trước -> đóng span cũ và mở span mới
                if curr_label is not None:
                    asp, sent = curr_label.split("#", 1) if "#" in curr_label else (curr_label, "neutral")
                    extracted_spans.append({
                        "start": curr_start,
                        "end": curr_end,
                        "aspect": asp,
                        "sentiment": sent,
                        "label": curr_label,
                    })
                curr_label = label
                curr_start = tok["start"]
                curr_end = tok["end"]
                
        else: # tag == "O"
            if curr_label is not None:
                asp, sent = curr_label.split("#", 1) if "#" in curr_label else (curr_label, "neutral")
                extracted_spans.append({
                    "start": curr_start,
                    "end": curr_end,
                    "aspect": asp,
                    "sentiment": sent,
                    "label": curr_label,
                })
                curr_label = None
                
    # Đóng span cuối cùng nếu còn
    if curr_label is not None:
        asp, sent = curr_label.split("#", 1) if "#" in curr_label else (curr_label, "neutral")
        extracted_spans.append({
            "start": curr_start,
            "end": curr_end,
            "aspect": asp,
            "sentiment": sent,
            "label": curr_label,
        })
        
    return extracted_spans


def extracted_spans_to_vector(extracted_spans: List[Dict[str, Any]]) -> List[int]:
    """
    Quy đổi danh sách spans được dự đoán (từ mô hình BIO/CRF) về vector 5 chiều (4 sentiments).
    Đồng thời áp dụng quy tắc trung bình/hòa hoãn cảm xúc nếu có nhiều spans cùng aspect.
    """
    aspect_sentiments: Dict[str, List[str]] = {asp: [] for asp in ASPECTS}
    for span in extracted_spans:
        asp = span["aspect"]
        sent = span["sentiment"]
        if asp in aspect_sentiments:
            aspect_sentiments[asp].append(sent)
            
    vector = []
    for asp in ASPECTS:
        final_sentiment = resolve_aspect_sentiment(aspect_sentiments[asp])
        vector.append(SENTIMENT_TO_ID[final_sentiment])
        
    return vector
