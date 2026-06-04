import os
import cv2
import torch
import numpy as np
import pandas as pd
import speech_recognition as sr
from moviepy.video.io.VideoFileClip import VideoFileClip
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tensorflow.keras.models import model_from_json
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

from fusion_utils import get_face_probs, get_text_probs, get_voice_probs
from load_data import extract_features
from model import Dual

# --- 1. CẤU HÌNH ---
WEIGHTS = {'face': 0.4, 'text': 0.4, 'voice': 0.2}
CLASS_LABELS = ["Tích cực", "Tiêu cực", "Trung tính"]
VIDEO_DIR = "C:/Users/HELLO/Downloads/DATN/data/data_video_test"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. KHỞI TẠO MÔ HÌNH ---
# Load Face (Keras)
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
json_file = open('C:/Users/HELLO/Downloads/DATN/code_emotion/cnn_new/emotion_detection.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
classifier = model_from_json(loaded_model_json)
classifier.load_weights('C:/Users/HELLO/Downloads/DATN/code_emotion/cnn_new/best_model.keras')

# Load Text (PhoBERT)
tokenizer = AutoTokenizer.from_pretrained("C:/Users/HELLO/Downloads/DATN/code_text/saved_model_text_new")
text_model = AutoModelForSequenceClassification.from_pretrained("C:/Users/HELLO/Downloads/DATN/code_text/saved_model_text_new").to(device).eval()

# Load Voice (Dual)
voice_model = Dual().to(device)
voice_model.load_state_dict(torch.load("C:/Users/HELLO/Downloads/DATN/code_voice/train_new/best_model_voice_by_vid.pth", map_location=device))
voice_model.eval()

# --- 3. VÒNG LẶP XỬ LÝ ---
results = []
recognizer = sr.Recognizer()

for root, dirs, files in os.walk(VIDEO_DIR):
    for file in files:
        if file.lower().endswith(('.mp4', '.avi', '.mov')):
            path = os.path.join(root, file)
            label_true = os.path.basename(root)
            print(f"Đang xử lý: {file}")

            try:
                # --- TRÍCH XUẤT DỮ LIỆU ĐẦU VÀO (FACE) ---
                cap = cv2.VideoCapture(path)
                frame_count = 0
                FRAME_SKIP = 100
                face_probs_list = []

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_count += 1
                    if frame_count % FRAME_SKIP != 0:
                        continue

                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
                    
                    if len(faces) > 0:
                        (x, y, w, h) = faces[0]
                        roi = cv2.resize(gray[y:y+h, x:x+w], (48, 48))
                        # Định dạng lại đầu vào cho đúng cấu trúc CNN xám (1, 48, 48, 1)
                        roi = np.expand_dims(np.expand_dims(roi.astype('float')/255.0, 0), -1)
                        
                        # Dự đoán xác suất khuôn mặt của frame này và lưu lại
                        p_face_single = get_face_probs(classifier, roi)
                        face_probs_list.append(p_face_single)

                cap.release()

                # Nếu tìm thấy mặt trong video, lấy trung bình cộng xác suất các frame
                if len(face_probs_list) > 0:
                    p_face = np.mean(face_probs_list, axis=0)
                else:
                    # Nếu cả video không tìm thấy mặt nào, trả về mảng xác suất bằng 0
                    p_face = np.zeros((1, 3))

                # Text: Trích xuất audio và nhận diện tiếng nói
                with VideoFileClip(path) as clip:
                    clip.audio.write_audiofile("temp.wav", codec='pcm_s16le', logger=None)
                with sr.AudioFile("temp.wav") as source:
                    audio = recognizer.record(source)
                    text_result = recognizer.recognize_google(audio, language="vi-VN")

                # Voice: Trích xuất đặc trưng
                mfccs = extract_features("temp.wav", n_mfcc=128, max_len=200)
                mfccs_tensor = torch.FloatTensor(mfccs).unsqueeze(0).unsqueeze(0).to(device)

                # --- GỌI HÀM FUSION ---
                # Đã lược bỏ dòng ghi đè p_face cũ để giữ nguyên giá trị trung bình cộng
                p_text = get_text_probs(text_model, tokenizer, text_result, device)
                p_voice = get_voice_probs(voice_model, mfccs_tensor, device)

                final_probs = (WEIGHTS['face']*p_face + WEIGHTS['text']*p_text + WEIGHTS['voice']*p_voice)
                pred_label = CLASS_LABELS[np.argmax(final_probs)]
                
                results.append({'true': label_true, 'pred': pred_label})
                print(f"   ↳ Kết quả: {pred_label}")
            except Exception as e:
                print(f"   ❌ Lỗi xử lý {file}: {e}")

# --- 4. XUẤT BÁO CÁO ---
df = pd.DataFrame(results)

# 1. Chuẩn hóa nhãn: Map tên thư mục về đúng định dạng label của mô hình
label_map = {
    'Tichcuc': 'Tích cực',
    'Tieucuc': 'Tiêu cực',
    'Trungtinh': 'Trung tính'
}

# Áp dụng chuẩn hóa
df['true_mapped'] = df['true'].map(label_map)

# 2. Định nghĩa thứ tự nhãn đồng nhất
labels_order = ["Tích cực", "Tiêu cực", "Trung tính"]

# 3. Tính toán báo cáo và ma trận nhầm lẫn
report = classification_report(df['true_mapped'], df['pred'], labels=labels_order, target_names=labels_order, zero_division=0)
cm = confusion_matrix(df['true_mapped'], df['pred'], labels=labels_order)

# 4. In và Lưu báo cáo vào file .txt đúng địa chỉ thư mục của dự án
output_txt_path = "C:/Users/HELLO/Downloads/DATN/danh_gia_fusion_new.txt"
with open(output_txt_path, 'w', encoding='utf-8') as f:
    f.write("📊 BÁO CÁO KẾT QUẢ FUSION 3 MÔ HÌNH\n")
    f.write("=" * 40 + "\n")
    f.write(report)
print(f"✅ Đã lưu báo cáo vào: {output_txt_path}")

# 5. Vẽ biểu đồ Confusion Matrix và lưu cùng thư mục
output_img_path = "C:/Users/HELLO/Downloads/DATN/danh_gia_fusion_new.png"
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=labels_order, yticklabels=labels_order)
plt.title("Confusion Matrix - Fusion Model")
plt.ylabel("Thực tế")
plt.xlabel("Dự đoán")
plt.savefig(output_img_path, dpi=300)
plt.close()
print(f"✅ Đã lưu ma trận nhầm lẫn vào: {output_img_path}")