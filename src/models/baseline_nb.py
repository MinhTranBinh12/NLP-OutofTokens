"""
Module Baseline Model cho bài toán Phân tích Cảm xúc Đa khía cạnh (ViCloABSA).
Sử dụng phương pháp Machine Learning truyền thống: TF-IDF + Naive Bayes (Multi-Output).

Nhánh: feat/baseline
"""

import sys
import os
from typing import List, Dict, Union, Optional
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, ComplementNB

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

try:
    from src.metrics import ASPECTS
except ImportError:
    ASPECTS = ["Material", "General", "Service", "Design", "Price"]

NUM_ASPECTS = len(ASPECTS)


class BaselineNaiveBayesModel:
    """
    Mô hình Baseline kết hợp TF-IDF Vectorizer và 5 bộ phân loại Naive Bayes độc lập
    tương ứng với 5 khía cạnh: Material, General, Service, Design, Price.
    
    Mỗi bộ phân loại dự đoán 1 trong 4 nhãn:
      0: None (Không đề cập)
      1: positive (Tích cực)
      2: neutral (Trung lập)
      3: negative (Tiêu cực)
    """

    def __init__(
        self,
        ngram_range: tuple = (1, 2),
        max_features: int = 10000,
        min_df: int = 2,
        sublinear_tf: bool = True,
        alpha: float = 0.1,
        nb_type: str = "multinomial",
    ):
        """
        Khởi tạo pipeline với cấu hình TF-IDF và Naive Bayes.
        
        Args:
            alpha: Hệ số làm mịn Laplace/Lidstone (mặc định 0.1 cho n-gram thưa).
            nb_type: 'multinomial' (MultinomialNB) hoặc 'complement' (ComplementNB).
        """
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.min_df = min_df
        self.sublinear_tf = sublinear_tf
        self.alpha = alpha
        self.nb_type = nb_type

        self.vectorizer = TfidfVectorizer(
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            min_df=self.min_df,
            sublinear_tf=self.sublinear_tf,
        )

        # 5 mô hình Naive Bayes độc lập cho 5 khía cạnh
        classifier_cls = ComplementNB if self.nb_type == "complement" else MultinomialNB
        self.classifiers: Dict[str, Union[MultinomialNB, ComplementNB]] = {
            aspect: classifier_cls(alpha=self.alpha)
            for aspect in ASPECTS
        }
        self.is_fitted = False

    def fit(self, texts: List[str], y: Union[np.ndarray, List[List[int]]]) -> "BaselineNaiveBayesModel":
        """
        Huấn luyện TF-IDF vectorizer và 5 mô hình Naive Bayes.
        
        Args:
            texts: Danh sách câu bình luận (N mẫu).
            y: Mảng 2D (N, 5) chứa nhãn số [0, 1, 2, 3] tương ứng 5 khía cạnh.
        """
        y_arr = np.array(y)
        if y_arr.shape[1] != NUM_ASPECTS:
            raise ValueError(
                f"Đầu ra y phải có kích thước (N, {NUM_ASPECTS}), hiện tại là {y_arr.shape}"
            )

        print(f"[BaselineNB - {self.nb_type}] Đang trích xuất đặc trưng TF-IDF...")
        X_tfidf = self.vectorizer.fit_transform(texts)
        print(f"[BaselineNB] TF-IDF shape: {X_tfidf.shape}")

        for i, aspect in enumerate(ASPECTS):
            y_aspect = y_arr[:, i]
            print(f"[BaselineNB] Đang huấn luyện Naive Bayes cho khía cạnh: {aspect}...")
            self.classifiers[aspect].fit(X_tfidf, y_aspect)

        self.is_fitted = True
        print("[BaselineNB] Hoàn tất quá trình huấn luyện.")
        return self

    def predict(self, texts: List[str]) -> np.ndarray:
        """
        Dự đoán nhãn vector 5 chiều cho danh sách câu bình luận đầu vào.
        
        Args:
            texts: Danh sách câu bình luận (N mẫu).
            
        Returns:
            np.ndarray có shape (N, 5), mỗi hàng là vector nhãn [0, 1, 2, 3]^5.
        """
        if not self.is_fitted:
            raise RuntimeError("Mô hình chưa được huấn luyện! Vui lòng gọi fit() trước.")

        X_tfidf = self.vectorizer.transform(texts)
        preds = []
        for aspect in ASPECTS:
            y_pred_aspect = self.classifiers[aspect].predict(X_tfidf)
            preds.append(y_pred_aspect)

        return np.column_stack(preds)

    def predict_proba(self, texts: List[str]) -> Dict[str, np.ndarray]:
        """
        Dự đoán xác suất các lớp cho 5 khía cạnh.
        """
        if not self.is_fitted:
            raise RuntimeError("Mô hình chưa được huấn luyện! Vui lòng gọi fit() trước.")

        X_tfidf = self.vectorizer.transform(texts)
        probas = {}
        for aspect in ASPECTS:
            probas[aspect] = self.classifiers[aspect].predict_proba(X_tfidf)
        return probas

    def save(self, filepath: str) -> None:
        """Lưu toàn bộ mô hình (vectorizer + classifiers) ra file bằng joblib."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)
        print(f"[BaselineNB] Đã lưu mô hình thành công tại: {filepath}")

    @staticmethod
    def load(filepath: str) -> "BaselineNaiveBayesModel":
        """Nạp mô hình đã lưu từ file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Không tìm thấy file mô hình: {filepath}")
        model = joblib.load(filepath)
        print(f"[BaselineNB] Đã nạp mô hình từ: {filepath}")
        return model

