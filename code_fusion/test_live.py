import os
import cv2
import torch
import numpy as np
import speech_recognition as sr
import time
import winsound 
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tensorflow.keras.models import model_from_json

from fusion_utils import get_face_probs, get_text_probs, get_voice_probs
from load_data import extract_features
from model import Dual

# --- 1. CẤU HÌNH ---
WEIGHTS = {'face': 0.4, 'text': 0.4, 'voice': 0.2}
CLASS_LABELS = ["Tích cực", "Tiêu cực", "Trung tính"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. KHỞI TẠO CÁC MÔ HÌNH AI ---
print("⏳ Đang nạp hệ thống mô hình AI đa phương thức...")

face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
with open('C:/Users/HELLO/Downloads/DATN/code_emotion/cnn_new/emotion_detection.json', 'r') as json_file:
    loaded_model_json = json_file.read()
classifier = model_from_json(loaded_model_json)
classifier.load_weights('C:/Users/HELLO/Downloads/DATN/code_emotion/cnn_new/best_model.keras')

tokenizer = AutoTokenizer.from_pretrained("C:/Users/HELLO/Downloads/DATN/code_text/saved_model_text_new")
text_model = AutoModelForSequenceClassification.from_pretrained("C:/Users/HELLO/Downloads/DATN/code_text/saved_model_text_new").to(device).eval()

voice_model = Dual().to(device)
voice_model.load_state_dict(torch.load("C:/Users/HELLO/Downloads/DATN/code_voice/train_new/best_model_voice_by_vid.pth", map_location=device))
voice_model.eval()

print("🎯 Toàn bộ mô hình đã sẵn sàng hoạt động!")

# --- 3. CẤU HÌNH THU ÂM NÂNG CAO (TĂNG ĐỘ NHẠY KHI NGỒI XA) ---
recognizer = sr.Recognizer()
microphone = sr.Microphone()

# Thiết lập thủ công các thuộc tính tăng cường độ nhạy bắt âm
recognizer.dynamic_energy_threshold = False  # Khóa tính năng tự điều chỉnh ngưỡng để tránh thuật toán tự tăng biên độ chặn khi im lặng
recognizer.energy_threshold = 150            # Hạ thấp ngưỡng bắt âm (Mặc định là 300). Số càng nhỏ mic càng nhạy với âm thanh ở xa
recognizer.pause_threshold = 1.2             # Đợi 1.2 giây im lặng mới tính là hết câu (tránh việc nói nhỏ bị ngắt quãng giữa chừng)

audio_data_holder = None

def audio_callback(recognizer_instance, audio):
    global audio_data_holder
    audio_data_holder = audio
    print("🎙️ [Audio] -> Đã ghi nhận xong một đoạn thoại ngầm...")

print("🎙️ [Audio] Hệ thống Microphone ĐỘ NHẠY CAO đã sẵn sàng nhận diện ngầm...")

# --- 4. LUỒNG CHÍNH: HIỂN THỊ CAMERA VÀ PHÂN TÍCH FUSION ---
def main_live_test():
    global audio_data_holder
    
    # 1. MỞ CAMERA TRƯỚC (Ưu tiên luồng đồ họa chính)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Không thể mở được Webcam!")
        return

    # Khởi tạo các biến cấu hình hiển thị và bỏ qua frame
    face_probs_list = []
    has_beeped = False  
    start_time = 0
    frame_count = 0
    FRAME_SKIP = 3  
    current_face_emotion = "Dang quet..."
    box_color = (0, 255, 0)
    stop_listening = None

    print("\n=======================================================")
    print("🎬 ĐANG KHỞI ĐỘNG ỐNG KÍNH CAMERA...")
    print("👉 CLICK CHUỘT VÀ O MÀN HÌNH CAM, rồi nhấn phím 'q' để DỪNG và xem KẾT QUẢ.")
    print("=======================================================\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        frame = cv2.flip(frame, 1) 
        h_img, w_img, _ = frame.shape
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)
        
        # --- ĐỒNG BỘ TIẾNG BEEP VÀ BẬT MIC KHI CAM ĐÃ LÊN HÌNH ỔN ĐỊNH ---
        if not has_beeped:
            # Hiện cửa sổ cam trước 1 frame để Windows nạp giao diện GUI
            cv2.imshow('He thong giam sat Cam xuc DATN (Nhan Q de dung)', frame)
            cv2.waitKey(100) # Delay nhẹ 100ms cho mượt cửa sổ
            
            # KÍCH HOẠT LẮNG NGHE NGẦM SAU KHI CAM ĐÃ LÊN HÌNH
            stop_listening = recognizer.listen_in_background(microphone, audio_callback, phrase_time_limit=10)
            
            # Phát tiếng BEEP chốt thời gian bắt đầu nói
            winsound.Beep(800, 400) 
            start_time = time.time() 
            has_beeped = True
            
        # Thông báo trực quan trên màn hình Camera
        if time.time() - start_time < 3.0:
            cv2.putText(frame, ">>> ET DA SAN SANG! HAY NOI NGAY <<<", (30, h_img - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, "He thong dang thu thap...", (30, h_img - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), box_color, 2)
            cv2.putText(frame, f"Mat: {current_face_emotion}", (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            
            # Giảm tải tính toán Face CNN để chống lag cam
            if frame_count % FRAME_SKIP == 0:
                roi = cv2.resize(gray[y:y+h, x:x+w], (48, 48))
                roi = np.expand_dims(np.expand_dims(roi.astype('float')/255.0, 0), -1)
                
                p_face_single = get_face_probs(classifier, roi)
                face_probs_list.append(p_face_single)
                
                label_map_no_sign = {
                    "Tích cực": "Tich cuc",
                    "Tiêu cực": "Tieu cuc",
                    "Trung tính": "Trung tinh"
                }
                raw_emotion = CLASS_LABELS[np.argmax(p_face_single)]
                current_face_emotion = label_map_no_sign.get(raw_emotion, raw_emotion)

        # Đẩy frame hình ảnh ra màn hình liên tục ở luồng xử lý chính
        cv2.imshow('He thong giam sat Cam xuc DATN (Nhan Q de dung)', frame)
        
        # Nhấn phím 'q' để kết thúc câu test
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n🛑 Đã nhận tín hiệu dừng! Vui lòng đợi 1.5 giây để đóng gói dữ liệu âm thanh...")
            time.sleep(1.5) 
            break

    # Thu dọn tài nguyên phần cứng sau khi vòng lặp dừng
    cap.release()
    cv2.destroyAllWindows()
    if stop_listening is not None:
        stop_listening(wait_for_stop=False)
    
    # --- 5. TỔNG HỢP VÀ TÍNH TOÁN KẾT QUẢ ---
    print("⏳ Bước 1: Trích xuất trung bình cộng đặc trưng Khuôn mặt...")
    if len(face_probs_list) > 0:
        p_face = np.mean(face_probs_list, axis=0).flatten() 
    else:
        p_face = np.zeros(3)

    print("⏳ Bước 2: Đang trích xuất đoạn âm thanh bạn vừa nói...")
    p_voice = np.zeros(3)
    p_text = np.zeros(3)
    temp_wav = "live_local_temp.wav"

    if audio_data_holder is not None:
        try:
            with open(temp_wav, "wb") as f:
                f.write(audio_data_holder.get_wav_data())
                
            # 2a. Tính toán đặc trưng giọng nói (Voice - MFCC)
            try:
                mfccs = extract_features(temp_wav, n_mfcc=128, max_len=200)
                mfccs_tensor = torch.FloatTensor(mfccs).unsqueeze(0).unsqueeze(0).to(device)
                p_voice_raw = get_voice_probs(voice_model, mfccs_tensor, device)
                p_voice = np.array(p_voice_raw).flatten() 
            except Exception as e:
                print(f"❌ Lỗi xử lý mô hình Voice: {e}")
                
            # 2b. Tính toán đặc trưng văn bản với khối Debug chi tiết
            try:
                with sr.AudioFile(temp_wav) as source_file:
                    audio_to_text = recognizer.record(source_file)
                
                text_result = recognizer.recognize_google(audio_to_text, language="vi-VN")
                print(f"📝 Lời thoại thu được: \"{text_result}\"")
                p_text_raw = get_text_probs(text_model, tokenizer, text_result, device)
                p_text = np.array(p_text_raw).flatten() 
            except sr.UnknownValueError:
                print("❌ [Lỗi Text]: Google không dịch được từ ngữ (Nói quá ngắn hoặc mic bị nhiễu).")
            except sr.RequestError as e:
                print(f"❌ [Lỗi Text]: Mất kết nối internet tới Google API; {e}")
            except Exception as e:
                print(f"❌ [Lỗi Text] Lỗi hệ thống khác khi xử lý PhoBERT: {e}")
                
            if os.path.exists(temp_wav):
                os.remove(temp_wav)
        except Exception as e:
            print(f"⚠️ Lỗi khi ghi file âm thanh tạm: {e}")
    else:
        print("⚠️ Không nhận được dữ liệu đoạn thoại hoàn chỉnh từ Microphone.")

    # --- 6. HÀM FUSION ĐA PHƯƠNG THỨC ---
    final_probs = (WEIGHTS['face'] * p_face + WEIGHTS['text'] * p_text + WEIGHTS['voice'] * p_voice)
    pred_idx = np.argmax(final_probs)
    final_emotion = CLASS_LABELS[pred_idx]

    # --- 7. IN BÁO CÁO VÀ GHI LOG FILE ---
    output_log_path = "C:/Users/HELLO/Downloads/DATN/ket_qua_test_truc_tiep.txt"
    report_str = f"""
======================================================
📊 KẾT QUẢ PHÂN TÍCH CẢM XÚC TRỰC TIẾP CHẠY LOCAL
======================================================
🏆 CẢM XÚC CHUNG (FUSION): {final_emotion.upper()}

[Chi tiết trọng số thành phần]:
- Khuôn mặt (Face): Tích cực: {p_face[0]*100:.2f}% | Tiêu cực: {p_face[1]*100:.2f}% | Trung tính: {p_face[2]*100:.2f}%
- Văn bản  (Text): Tích cực: {p_text[0]*100:.2f}% | Tiêu cực: {p_text[1]*100:.2f}% | Trung tính: {p_text[2]*100:.2f}%
- Giọng nói (Voice): Tích cực: {p_voice[0]*100:.2f}% | Tiêu cực: {p_voice[1]*100:.2f}% | Trung tính: {p_voice[2]*100:.2f}%
------------------------------------------------------
🔥 Xác suất Fusion cuối cùng:
- TÍCH CỰC: {final_probs[0]*100:.2f}%
- TIÊU CỰC: {final_probs[1]*100:.2f}%
- TRUNG TÍNH: {final_probs[2]*100:.2f}%
======================================================
"""
    print(report_str)
    try:
        with open(output_log_path, "a", encoding="utf-8") as f:
            f.write(f"\n[Test lúc: {time.strftime('%Y-%m-%d %H:%M:%S')}]" + report_str)
        print(f"✅ Đã lưu báo cáo chi tiết vào file: {output_log_path}")
    except Exception as e:
        print(f"❌ Không thể ghi file log: {e}")

if __name__ == "__main__":
    main_live_test()