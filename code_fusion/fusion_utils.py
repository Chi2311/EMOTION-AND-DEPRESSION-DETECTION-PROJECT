# Hàm trích xuất xác suất
# Đây là file giúp bạn lấy ra mảng xác suất [P_Tichcuc, P_Tieucuc, P_Trungtinh].
import numpy as np
import torch
import torch.nn.functional as F

# Định nghĩa thứ tự nhãn đồng nhất cho cả 3 mô hình
# PHẢI ĐẢM BẢO: Index 0=Tích cực, 1=Tiêu cực, 2=Trung tính
LABELS = ["Tích cực", "Tiêu cực", "Trung tính"]

def get_face_probs(classifier, roi):
    """Lấy xác suất từ model Keras (Face)"""
    # roi đã được tiền xử lý thành dạng (1, 48, 48, 1) hoặc tương đương
    prediction = classifier.predict(roi, verbose=0)[0]
    return prediction # Trả về mảng 3 giá trị

def get_text_probs(model, tokenizer, text, device):
    """Lấy xác suất từ model PhoBERT (Text)"""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = F.softmax(outputs.logits, dim=-1).detach().cpu().numpy()[0]
    return probs

def get_voice_probs(model, mfccs_tensor, device):
    """Lấy xác suất từ model Dual (Voice)"""
    with torch.no_grad():
        output = model(mfccs_tensor)
    probs = F.softmax(output, dim=1).detach().cpu().numpy()[0]
    return probs