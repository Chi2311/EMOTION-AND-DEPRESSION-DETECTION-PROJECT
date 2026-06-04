import os
import sys
import cv2
import torch
import numpy as np
import speech_recognition as sr
import subprocess
import time
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import ImageFont, ImageDraw, Image

# Import các hàm đặc trưng từ đồ án của bạn
from fusion_utils import get_face_probs, get_text_probs, get_voice_probs
from load_data import extract_features
from model import Dual

# --- CẤU HÌNH HỆ THỐNG TIẾNG VIỆT CHO TERMINAL ---
if os.name == 'nt':
    subprocess.run(['chcp', '65001'], shell=True, capture_output=True)
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# --- 1. CẤU HÌNH TRỌNG SỐ FUSION ---
WEIGHTS = {'face': 0.4, 'text': 0.4, 'voice': 0.2}
CLASS_LABELS = ["Tích cực", "Tiêu cực", "Trung tính"]
class_labels_en = ['Tichcuc', 'Tieucuc', 'Trungtinh'] # Dùng để map màu sắc từ code cũ
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

color_dict = {
    'Tichcuc': (0, 255, 0),     # Xanh lá cây
    'Tieucuc': (0, 0, 255),     # Đỏ
    'Trungtinh': (0, 255, 255)  # Màu Vàng
}

try:
    font = ImageFont.truetype("arial.ttf", 32)
except:
    font = ImageFont.load_default()

# --- 2. KHỞI TẠO CÁC MÔ HÌNH AI ---
print("⏳ Đang nạp hệ thống mô hình AI đa phương thức...")

# Load Face (Keras) từ đường dẫn mới của bạn
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
with open('C:/Users/HELLO/Downloads/DATN/code_emotion/cnn_new/emotion_detection.json', 'r') as json_file:
    loaded_model_json = json_file.read()
classifier = model_from_json(loaded_model_json)
classifier.load_weights('C:/Users/HELLO/Downloads/DATN/code_emotion/cnn_new/best_model.keras')

# Load Text (PhoBERT)
tokenizer = AutoTokenizer.from_pretrained("C:/Users/HELLO/Downloads/DATN/code_text/saved_model_text_new")
text_model = AutoModelForSequenceClassification.from_pretrained("C:/Users/HELLO/Downloads/DATN/code_text/saved_model_text_new").to(device).eval()

# Load Voice (Dual)
voice_model = Dual().to(device)
voice_model.load_state_dict(torch.load("C:/Users/HELLO/Downloads/DATN/code_voice/train_new/best_model_voice_by_vid.pth", map_location=device))
voice_model.eval()

print("🎯 Toàn bộ mô hình đã sẵn sàng hoạt động!")

# --- 3. KHỞI TẠO MICROPHONE VÀ THÍCH NGHI NHIỄU (2 GIÂY) ---
r = sr.Recognizer()
microphone = sr.Microphone()

print("\n" + "="*45)
print("      CHẾ ĐỘ LẮNG NGHE THÔNG MINH (FIX NHIỄU)")
print("=======================================================")
with microphone as source:
    print("[Trạng thái]: Đang thích nghi tiếng ồn môi trường (2s)... Vui lòng giữ im lặng.")
    r.adjust_for_ambient_noise(source, duration=2)
print("🎙️  Hệ thống Microphone đã sẵn sàng nhận diện!")

# Biến lưu trữ dữ liệu hình ảnh/âm thanh toàn cục
audio_data_holder = None
face_probs_list = []
sentence_count = 1

# Hàm callback xử lý âm thanh ngầm khi phát hiện câu nói kết thúc
def audio_callback(recognizer_instance, audio):
    global audio_data_holder
    audio_data_holder = audio

# --- 4. LUỒNG CHÍNH CHẠY CAMERA VÀ FUSION ---
def main_live_fusion():
    global audio_data_holder, face_probs_list, sentence_count
    
    # Bật luồng thu âm ngầm (Phrase time limit 10 giây để tránh tràn câu)
    stop_listening = r.listen_in_background(microphone, audio_callback, phrase_time_limit=10)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Không thể mở được Webcam!")
        stop_listening(wait_for_stop=False)
        return

    print("\n=======================================================")
    print("🎬 CAMERA ĐÃ BẬT TRỰC TIẾP!")
    print("👉 Nhìn vào Cam và nói câu tiếng Việt bất kỳ.")
    print("👉 Nói các từ 'kết thúc', 'thoát', 'dừng lại' hoặc bấm 'q' để đóng hệ thống.")
    print("=======================================================\n")

    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Tự động kiểm tra thời gian chờ 60 giây im lặng giống code cũ
        elapsed_time = time.time() - start_time
        if elapsed_time > 60:
            print(f"\n[Thông báo]: Đã quá 60 giây không phát hiện câu nói hợp lệ. Tự động dừng.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            
            # Chuẩn hóa ảnh cho mô hình Face
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            # Dự đoán cảm xúc khuôn mặt từ frame hiện tại
            preds = classifier.predict(roi, verbose=0)[0]
            face_probs_list.append(preds) # Tích lũy xác suất khuôn mặt
            
            label_en = class_labels_en[preds.argmax()]
            label_vi = CLASS_LABELS[preds.argmax()]
            box_color = color_dict[label_en]

            # Vẽ khung hình chữ nhật quanh mặt
            cv2.rectangle(frame, (x, y), (x+w, y+h), box_color, 2)

            # Viết chữ tiếng Việt chuẩn font bằng PIL (Đảo màu BGR sang RGB cho PIL vẽ đúng màu)
            img_pil = Image.fromarray(frame)
            draw = ImageDraw.Draw(img_pil)
            draw.text((x, y - 40), label_vi, font=font, fill=(box_color[2], box_color[1], box_color[0]))
            frame = np.array(img_pil)

        # Hiển thị camera trực tiếp lên cửa sổ đồ họa
        cv2.imshow('Emotion Detection - Bam Q de thoat', frame)

        # --- KIỂM TRA VÀ XỬ LÝ KHI CÓ DỮ LIỆU ÂM THANH MỚI ĐỔ VỀ ---
        if audio_data_holder is not None:
            current_audio = audio_data_holder
            audio_data_holder = None # Reset ngay để tránh xử lý trùng lặp
            
            print("\n--- 🎧 Đang trích xuất dữ liệu và xử lý thuật toán Fusion ---")
            temp_wav = "live_fusion_temp.wav"
            
            p_voice = np.zeros(3)
            p_text = np.zeros(3)
            text_result = ""
            
            try:
                # 1. Lưu file audio tạm thời để chạy mô hình Voice (MFCC)
                with open(temp_wav, "wb") as f:
                    f.write(current_audio.get_wav_data())
                
                # Tính toán Đặc trưng tông giọng (Voice)
                try:
                    mfccs = extract_features(temp_wav, n_mfcc=128, max_len=200)
                    mfccs_tensor = torch.FloatTensor(mfccs).unsqueeze(0).unsqueeze(0).to(device)
                    p_voice_raw = get_voice_probs(voice_model, mfccs_tensor, device)
                    p_voice = np.array(p_voice_raw).flatten()
                except Exception as e:
                    print(f" ❌ Lỗi xử lý mô hình Tông Giọng: {e}")

                # 2. Chuyển giọng nói thành chữ bằng Google API (Text)
                try:
                    text_result = r.recognize_google(current_audio, language='vi-VN')
                    print(f">> Bạn nói: {text_result}")
                    
                    # Tính toán Đặc trưng văn bản (Text PhoBERT)
                    p_text_raw = get_text_probs(text_model, tokenizer, text_result, device)
                    p_text = np.array(p_text_raw).flatten()
                except sr.UnknownValueError:
                    print(" -> (Không rõ âm thanh/Tiếng nhiễu...)")
                except Exception as e:
                    print(f" ❌ Lỗi mô hình văn bản PhoBERT: {e}")

                # 3. Tính toán Đặc trưng Khuôn mặt trung bình thu thập được trong lúc nói
                if len(face_probs_list) > 0:
                    p_face = np.mean(face_probs_list, axis=0).flatten()
                else:
                    p_face = np.zeros(3)
                
                # Reset mảng khuôn mặt chuẩn bị cho câu nói tiếp theo
                face_probs_list = []

                # 4. TÍNH TOÁN FUSION ĐA PHƯƠNG THỨC CHUNG CHÍNH XÁC
                final_probs = (WEIGHTS['face'] * p_face + WEIGHTS['text'] * p_text + WEIGHTS['voice'] * p_voice)
                pred_idx = np.argmax(final_probs)
                final_emotion = CLASS_LABELS[pred_idx]

                # --- LƯU LẠI FILE VÀ IN BÁO CÁO RA TERMINAL VỚI TIẾNG VIỆT ĐẸP ---
                print(f" -> [Hệ thống]: Đã lưu kết quả phân tích câu số {sentence_count}")
                with open("text.txt", "a", encoding="utf-8") as f:
                    f.write(f"Câu {sentence_count}: {text_result if text_result else '(Không nhận diện được chữ)'}\n")
                    f.write(f"🏆 CẢM XÚC CHUNG (FUSION): {final_emotion}\n")
                    f.write(f"  + Face : Tích cực: {p_face[0]*100:.1f}% | Tiêu cực: {p_face[1]*100:.1f}% | Trung tính: {p_face[2]*100:.1f}%\n")
                    f.write(f"  + Text : Tích cực: {p_text[0]*100:.1f}% | Tiêu cực: {p_text[1]*100:.1f}% | Trung tính: {p_text[2]*100:.1f}%\n")
                    f.write(f"  + Voice: Tích cực: {p_voice[0]*100:.1f}% | Tiêu cực: {p_voice[1]*100:.1f}% | Trung tính: {p_voice[2]*100:.1f}%\n")
                    f.write("-" * 45 + "\n")
                
                # In bảng thống kê chi tiết ra màn hình
                print(f"\n=================== KẾT QUẢ CÂU SỐ {sentence_count} ===================")
                print(f"🏆 CẢM XÚC CHUNG TỔNG HỢP (FUSION): {final_emotion.upper()}")
                print(f"🤖 [Face]  -> Tích cực: {p_face[0]*100:.1f}% | Tiêu cực: {p_face[1]*100:.1f}% | Trung tính: {p_face[2]*100:.1f}%")
                print(f"📝 [Text]  -> Tích cực: {p_text[0]*100:.1f}% | Tiêu cực: {p_text[1]*100:.1f}% | Trung tính: {p_text[2]*100:.1f}%")
                print(f"🎙️  [Voice] -> Tích cực: {p_voice[0]*100:.1f}% | Tiêu cực: {p_voice[1]*100:.1f}% | Trung tính: {p_voice[2]*100:.1f}%")
                print(f"==========================================================\n")
                
                sentence_count += 1
                
                # KÍCH HOẠT LỆNH THOÁT BẰNG GIỌNG NÓI KHỚP CODE CŨ
                if text_result:
                    lower_text = text_result.lower()
                    if any(word in lower_text for word in ["kết thúc", "thoát", "dừng lại"]):
                        print("\n[Hệ thống]: Đã nhận lệnh dừng bằng giọng nói.")
                        if os.path.exists(temp_wav):
                            os.remove(temp_wav)
                        break

                # Đặt lại đồng hồ đếm thời gian im lặng
                start_time = time.time()

                if os.path.exists(temp_wav):
                    os.remove(temp_wav)

            except Exception as e:
                print(f"Lỗi hệ thống trong vòng lặp Fusion: {e}")

        # Nhấn nút 'q' đồ họa để thoát chủ động
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n🛑 Đã nhận lệnh thoát từ bàn phím.")
            break

    # Giải phóng thiết bị phần cứng
    cap.release()
    cv2.destroyAllWindows()
    stop_listening(wait_for_stop=False)
    print(" Hệ thống demo fusion trực tiếp đã đóng hoàn toàn.")

if __name__ == "__main__":
    main_live_fusion()