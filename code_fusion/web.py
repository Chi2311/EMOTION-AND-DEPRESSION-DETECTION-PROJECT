import os
# Ép chạy lõi Keras cũ để đọc file .json trơn tru trên Python 3.12
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import cv2
import torch
import numpy as np
import speech_recognition as sr
import winsound
import time
import sounddevice as sd
import ffmpeg  
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tensorflow.keras.models import model_from_json
import streamlit as st

# --- CẤU HÌNH GIAO DIỆN STREAMLIT ---
st.set_page_config(page_title="Nhận diện Cảm xúc Đa phương thức", layout="wide")
st.title("🎬 Hệ thống Thu hình & Thu âm Đồng thời (Fusion Model)")
st.subheader("Nhận diện Cảm xúc trực tiếp qua OpenCV & Microphone")

# --- HÀM LƯU DỮ LIỆU VÀO MYSQL LOCALHOST XAMPP ---
def save_to_localhost(text, emotion, p_face, p_text, p_voice, final_probs):
    try:
        import mysql.connector
        conn = mysql.connector.connect(
            host="localhost", user="root", password="", database="datn_emotion"
        )
        cursor = conn.cursor()
        sql = """INSERT INTO ket_qua_fusion (loi_thoai, cam_xuc_chug, pt_face, pt_text, pt_voice, pt_fusion) 
                 VALUES (%s, %s, %s, %s, %s, %s)"""
                 
        str_face = f"Tích cực: {p_face[0]*100:.2f}% | Tiêu cực: {p_face[1]*100:.2f}% | Trung tính: {p_face[2]*100:.2f}%"
        str_text = f"Tích cực: {p_text[0]*100:.2f}% | Tiêu cực: {p_text[1]*100:.2f}% | Trung tính: {p_text[2]*100:.2f}%"
        str_voice = f"Tích cực: {p_voice[0]*100:.2f}% | Tiêu cực: {p_voice[1]*100:.2f}% | Trung tính: {p_voice[2]*100:.2f}%"
        str_fusion = f"Tích cực: {final_probs[0]*100:.2f}% | Tiêu cực: {final_probs[1]*100:.2f}% | Trung tính: {final_probs[2]*100:.2f}%"
        
        cursor.execute(sql, (text, emotion, str_face, str_text, str_voice, str_fusion))
        conn.commit()
        cursor.close()
        conn.close()
        return True
    except Exception:
        return False

# --- 1. KHỞI TẠO VÀ CACHE MÔ HÌNH ---
@st.cache_resource(show_spinner=False)
def load_all_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    try:
        json_path = 'C:/Users/HELLO/Downloads/DATN/code_emotion/cnn_new/emotion_detection.json'
        weight_path = 'C:/Users/HELLO/Downloads/DATN/code_emotion/cnn_new/best_model.keras'
        with open(json_path, 'r') as json_file:
            loaded_model_json = json_file.read()
        classifier = model_from_json(loaded_model_json)
        classifier.load_weights(weight_path)
    except Exception:
        classifier = None

    try:
        text_model_path = "C:/Users/HELLO/Downloads/DATN/code_text/saved_model_text_new"
        tokenizer = AutoTokenizer.from_pretrained(text_model_path)
        text_model = AutoModelForSequenceClassification.from_pretrained(text_model_path).to(device).eval()
    except Exception:
        tokenizer = None
        text_model = None

    try:
        voice_weight_path = "C:/Users/HELLO/Downloads/DATN/code_voice/train_new/best_model_voice_by_vid.pth"
        from model import Dual 
        voice_model = Dual().to(device)
        voice_model.load_state_dict(torch.load(voice_weight_path, map_location=device))
        voice_model.eval()
    except Exception:
        voice_model = None
    
    return face_classifier, classifier, tokenizer, text_model, voice_model, device

if "recognizer" not in st.session_state:
    st.session_state.recognizer = sr.Recognizer()

WEIGHTS = {'face': 0.4, 'text': 0.4, 'voice': 0.2}
CLASS_LABELS = ["Tích cực", "Tiêu cực", "Trung tính"]

if "models_loaded" not in st.session_state:
    with st.spinner("⏳ Đang nạp hệ thống AI đa phương thức từ bộ nhớ máy tính..."):
        res = load_all_models()
        st.session_state.face_classifier, st.session_state.classifier, st.session_state.tokenizer, st.session_state.text_model, st.session_state.voice_model, st.session_state.device = res
        st.session_state.models_loaded = True

# --- KHỞI TẠO CÁC BIẾN TRẠNG THÁI (STATE) ĐIỀU KHIỂN LUỒNG ---
if "camera_on" not in st.session_state:
    st.session_state.camera_on = False
if "recording" not in st.session_state:
    st.session_state.recording = False
if "trigger_analysis" not in st.session_state:
    st.session_state.trigger_analysis = False
if "audio_frames_list" not in st.session_state:
    st.session_state.audio_frames_list = []
if "face_probs_list" not in st.session_state:
    st.session_state.face_probs_list = []

# --- 2. BỐ CỤC GIAO DIỆN WEB ---
col1, col2 = st.columns([1.2, 1])

with col1:
    st.markdown("### 📹 Khu vực tương tác Camera & Microphone")
    image_placeholder = st.empty()
    
    st.write("")
    b1, b2, b3 = st.columns([1, 1, 1])
    
    with b1:
        if not st.session_state.camera_on:
            if st.button("📷 BẬT CAMERA", use_container_width=True, type="primary"):
                st.session_state.camera_on = True
                st.session_state.trigger_analysis = False
                st.rerun()
        else:
            if st.button("📷 TẮT CAMERA", use_container_width=True):
                st.session_state.camera_on = False
                st.session_state.recording = False
                image_placeholder.empty()
                st.rerun()

    with b2:
        start_disabled = not st.session_state.camera_on or st.session_state.recording
        if st.button("🔴 BẮT ĐẦU THU (START)", use_container_width=True, disabled=start_disabled):
            st.session_state.recording = True
            st.session_state.audio_frames_list = []
            st.session_state.face_probs_list = []
            st.session_state.trigger_analysis = False
            
            # Khởi động Microphone thu âm tần số 16000Hz chuẩn Speech AI
            st.session_state.audio_stream = sd.InputStream(samplerate=16000, channels=1, dtype='int16')
            st.session_state.audio_stream.start()
            winsound.Beep(800, 400) 

    with b3:
        stop_disabled = not st.session_state.recording
        if st.button("⬜ DỪNG & PHÂN TÍCH (STOP)", use_container_width=True, disabled=stop_disabled):
            st.session_state.recording = False
            
            # Ngắt luồng Microphone và lấy toàn bộ dữ liệu âm thanh còn lại
            if "audio_stream" in st.session_state and st.session_state.audio_stream is not None:
                try:
                    data, _ = st.session_state.audio_stream.read(st.session_state.audio_stream.read_available())
                    if len(data) > 0:
                        st.session_state.audio_frames_list.append(data)
                    st.session_state.audio_stream.stop()
                    st.session_state.audio_stream.close()
                except Exception: pass
                    
            winsound.Beep(600, 400) 
            st.session_state.trigger_analysis = True
            st.rerun()

    # --- VÒNG LẶP ĐỒNG BỘ HIỂN THỊ CAMERA CHÍNH ---
    if st.session_state.camera_on:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) 
        frame_count = 0
        FRAME_SKIP = 3
        current_face_emotion = "Dang quet..."
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_writer = None
        
        while st.session_state.camera_on:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            frame = cv2.flip(frame, 1)
            
            if st.session_state.recording:
                if video_writer is None:
                    video_writer = cv2.VideoWriter('live_temp_video.avi', cv2.VideoWriter_fourcc(*'XVID'), 20.0, (width, height))
                video_writer.write(frame)
            else:
                if video_writer is not None:
                    video_writer.release()
                    video_writer = None

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = st.session_state.face_classifier.detectMultiScale(gray, 1.3, 5)
            
            display_frame = frame.copy()
            for (x, y, w, h) in faces:
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                roi = cv2.resize(gray[y:y+h, x:x+w], (48, 48))
                roi_norm = np.expand_dims(np.expand_dims(roi.astype('float')/255.0, 0), -1)
                
                if frame_count % FRAME_SKIP == 0:
                    try:
                        from fusion_utils import get_face_probs
                        p_face_single = get_face_probs(st.session_state.classifier, roi_norm)
                        
                        if st.session_state.recording:
                            st.session_state.face_probs_list.append(p_face_single)
                        
                        raw_emotion = CLASS_LABELS[np.argmax(p_face_single)]
                        label_map_no_sign = {"Tích cực": "Tich cuc", "Tiêu cực": "Tieu cuc", "Trung tính": "Trung tinh"}
                        current_face_emotion = label_map_no_sign.get(raw_emotion, raw_emotion)
                    except Exception: pass
                
                cv2.putText(display_frame, f"Mat: {current_face_emotion}", (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            
            if st.session_state.recording:
                cv2.putText(display_frame, "● RECORDING ACTIVE (MIC WORKING)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                if "audio_stream" in st.session_state and st.session_state.audio_stream is not None:
                    try:
                        avail = st.session_state.audio_stream.read_available()
                        if avail > 0:
                            data, _ = st.session_state.audio_stream.read(avail)
                            st.session_state.audio_frames_list.append(data)
                    except Exception: pass
            else:
                cv2.putText(display_frame, "STATUS: CAMERA READY (IDLE)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
            frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            image_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
            
            time.sleep(0.025)
            
            if not st.session_state.camera_on or st.session_state.trigger_analysis:
                break
                
        if video_writer is not None:
            video_writer.release()
        cap.release()

with col2:
    st.markdown("### 📊 Kết quả Phân tích Tổng hợp (Fusion)")
    
    if st.session_state.get("trigger_analysis", False):
        st.session_state.trigger_analysis = False
        
        with st.spinner("🧠 Hệ thống đang xử lý thuật toán AI và đồng bộ Video Preview..."):
            # ĐƯỢC ĐƯA LÊN ĐẦU KHỐI: Sửa lỗi NameError hoàn toàn
            temp_wav_path = "live_combined_audio.wav" 
            
            p_face = np.zeros(3)
            p_voice = np.zeros(3)
            p_text = np.zeros(3)
            text_result = "Không nhận diện được lời thoại"
            
            # 1. Trích xuất đặc trưng khuôn mặt
            if len(st.session_state.face_probs_list) > 0:
                p_face = np.mean(st.session_state.face_probs_list, axis=0).flatten()
                
            # 2. Xử lý lưu dữ liệu âm thanh từ RAM thành file WAVE
            if len(st.session_state.audio_frames_list) > 0:
                try:
                    combined_audio = np.concatenate(st.session_state.audio_frames_list, axis=0)
                    audio_bytes = combined_audio.tobytes()
                    
                    import wave
                    with wave.open(temp_wav_path, 'wb') as wav_file:
                        wav_file.setnchannels(1)       
                        wav_file.setsampwidth(2)       
                        wav_file.setframerate(16000)   
                        wav_file.writeframes(audio_bytes)
                    
                    if os.path.exists(temp_wav_path) and os.path.getsize(temp_wav_path) > 44:
                        # 2a. Tính toán đặc trưng giọng nói (Voice)
                        try:
                            from load_data import extract_features
                            from fusion_utils import get_voice_probs
                            mfccs = extract_features(temp_wav_path, n_mfcc=128, max_len=200)
                            mfccs_tensor = torch.FloatTensor(mfccs).unsqueeze(0).unsqueeze(0).to(st.session_state.device)
                            p_voice = np.array(get_voice_probs(st.session_state.voice_model, mfccs_tensor, st.session_state.device)).flatten()
                        except Exception: pass
                        
                        # 2b. Chuyển đổi giọng nói thành văn bản (Text)
                        try:
                            from fusion_utils import get_text_probs
                            with sr.AudioFile(temp_wav_path) as source:
                                audio_recorded = st.session_state.recognizer.record(source)
                            text_result = st.session_state.recognizer.recognize_google(audio_recorded, language="vi-VN")
                            
                            if text_result.strip():
                                p_text = np.array(get_text_probs(st.session_state.text_model, st.session_state.tokenizer, text_result, st.session_state.device)).flatten()
                        except Exception: pass
                except Exception: pass
            else:
                st.warning("🎙️ Không nhận diện được cấu trúc tệp âm thanh. Hãy kiểm tra lại kết nối thiết bị Microphone!")
            
            # --- ĐỒNG BỘ MUXING GỘP AUDIO VÀ VIDEO BẰNG FFMPEG ---
            if os.path.exists('live_temp_video.avi') and os.path.exists(temp_wav_path):
                try:
                    video_input = ffmpeg.input('live_temp_video.avi')
                    audio_input = ffmpeg.input(temp_wav_path)
                    # Sử dụng vcodec='libx264' và acodec='aac' để trình duyệt web hiển thị trực tiếp được
                    ffmpeg.output(video_input, audio_input, 'final_preview.mp4', vcodec='libx264', acodec='aac').run(overwrite_output=True, quiet=True)
                except Exception: pass

            if np.sum(p_face) == 0: p_face = np.array([0.33, 0.33, 0.33])
            if np.sum(p_text) == 0: p_text = np.array([0.33, 0.33, 0.33])
            if np.sum(p_voice) == 0: p_voice = np.array([0.33, 0.33, 0.33])

            # 3. Thuật toán kết hợp Fusion Trọng số
            final_probs = (WEIGHTS['face'] * p_face + WEIGHTS['text'] * p_text + WEIGHTS['voice'] * p_voice)
            pred_idx = np.argmax(final_probs)
            final_emotion = CLASS_LABELS[pred_idx]

            # 4. Lưu kết quả vào Localhost MySQL
            save_to_localhost(text_result, final_emotion, p_face, p_text, p_voice, final_probs)
            
            st.markdown(f"## 🏆 Cảm xúc nhận diện: **{final_emotion.upper()}**")
            st.info(f"📝 **Lời thoại ghi nhận:** \"{text_result}\"")
            
            report_data = {
                "Cảm xúc": CLASS_LABELS,
                "Khuôn mặt (40%)": [f"{p_face[0]*100:.2f}%", f"{p_face[1]*100:.2f}%", f"{p_face[2]*100:.2f}%"],
                "Lời thoại (40%)": [f"{p_text[0]*100:.2f}%", f"{p_text[1]*100:.2f}%", f"{p_text[2]*100:.2f}%"],
                "Giọng nói (20%)": [f"{p_voice[0]*100:.2f}%", f"{p_voice[1]*100:.2f}%", f"{p_voice[2]*100:.2f}%"],
                "Tổng hợp Fusion": [f"{final_probs[0]*100:.2f}%", f"{final_probs[1]*100:.2f}%", f"{final_probs[2]*100:.2f}%"]
            }
            st.table(report_data)
            
            st.rerun()

    # KHỐI HIỂN THỊ VIDEO LUÔN ĐƯỢC ĐẶT Ở NGOÀI CÙNG VÀ DƯỚI CÙNG CỦA COL2
    if os.path.exists('final_preview.mp4'):
        st.markdown("---")
        st.markdown("#### 🎬 Video Xem lại tương tác (Preview có âm thanh):")
        st.video('final_preview.mp4')
    else:
        st.caption("Chưa có dữ liệu kiểm tra. Hãy nhấn nút **📷 BẬT CAMERA** -> nhấn tiếp **🔴 BẮT ĐẦU THU (START)** để tạo phiên ghi mới.")