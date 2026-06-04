import numpy as np
import librosa
import os
from torch.utils.data import DataLoader, Dataset

n_mfcc = 128
window_size = 2048
strides = 512
window_size_stft = 1024
window = np.hanning(window_size_stft)

def load_emodata(link, sr = 16000, duration = 5):
    # Load the audio file
    wave_form, _ = librosa.load(path=link, sr=sr)
    # label = father folder name
    labels = os.path.basename(os.path.dirname(link))

    if len(wave_form) < sr * duration:
        # zero pad the audio if its less than 5 seconds
        wave_form = np.pad(wave_form, (0, sr * duration - len(wave_form)), 'symmetric')
        mfcc1 = librosa.feature.mfcc(y=y, sr=16000, n_mfcc=128, n_fft=2048, hop_length=512)
        # stft1 = librosa.core.spectrum.stft(wave_form, n_fft=window_size_stft, hop_length=256, window=window)
        # spect1 = 2 * np.abs(stft1) / np.sum(window)
        return np.array([mfcc1]), np.array([labels])

    elif len(wave_form) == sr * duration:
        mfcc2 = librosa.feature.mfcc(y=y, sr=16000, n_mfcc=128, n_fft=2048, hop_length=512)
        # stft2 = librosa.core.spectrum.stft(wave_form, n_fft=window_size_stft, hop_length=256, window=window)
        # spect2 = 2 * np.abs(stft2) / np.sum(window)
        # return the audio as it is
        return np.array([mfcc2]), np.array([labels])


    else:
        wave_segments = []
        _labels = [labels] * (len(wave_form) // (sr * duration) + 1)
        for i in range(0, len(wave_form), sr * duration):
            wave_segments.append(wave_form[i:i + sr * duration])

        # If the last segment is less than 5 seconds, then pad it with the last 5 seconds of the second last segment
        len_wave_segments_last = len(wave_segments[-1])
        padding = sr * duration - len_wave_segments_last
        temp = np.append(wave_segments[-2][sr * duration - padding:], wave_segments[-1])
        wave_segments[-1] = temp

        mfcc_seg = librosa.feature.mfcc(y=y, sr=16000, n_mfcc=128, n_fft=2048, hop_length=512)
        # stft_seg = librosa.core.spectrum.stft(wave_segments, n_fft=window_size_stft, hop_length=256, window=window)
        # spect_seg = 2 * np.abs(stft_seg) / np.sum(window)

        return np.array(mfcc_seg) , np.array(_labels)

#CODE LOAD DỮ LIỆU (GỐC)
# def load_data(data_folder):
#     # waves = []
#     mfccs = []
#     labels = []
#     for root, dirs, files in os.walk(data_folder):
#         for file in files:
#             if file.endswith(".wav"):
#                 link = os.path.join(root, file)
#                 mfcc, label = load_emodata(link)

#                 # waves.extend(wave)
#                 mfccs.extend(mfcc)
#                 labels.extend(label)

#     return np.array(mfccs), np.array(labels)

#CODE TEST

def extract_features(file_path, n_mfcc=128, max_len=200): # Cập nhật mặc định là 128
    try:
        y, sr = librosa.load(file_path, sr=16000) # Ép SR=16000
        mfcc = librosa.feature.mfcc(
            y=y, 
            sr=16000, 
            n_mfcc=n_mfcc, # Sẽ lấy giá trị 128
            n_fft=2048, 
            hop_length=512
        )
        
        # Padding hoặc cắt cho cùng độ dài
        if mfcc.shape[1] < max_len:
            pad_width = max_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0,0),(0,pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :max_len]
        return mfcc
    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return None

def load_dataset(data_folder):
    mfccs = []
    labels = []
    for idx, class_name in enumerate(sorted(os.listdir(data_folder))):
        class_path = os.path.join(data_folder, class_name)
        if not os.path.isdir(class_path):
            continue
        for file_name in os.listdir(class_path):
            if file_name.lower().endswith(".wav"):
                file_path = os.path.join(class_path, file_name)
                features = extract_features(file_path)
                if features is not None:
                    mfccs.append(features)
                    labels.append(idx)

    return np.array(mfccs), np.array(labels)
#########################

class EmoDataset(Dataset):
    def __init__(self, mfccs, labels):

        self.mfccs = mfccs
        self.labels = labels

    def __len__(self):
        return len(self.mfccs)

    def __getitem__(self, index):
        return self.mfccs[index], self.labels[index]

# import os
# import librosa
# import numpy as np

# def extract_features(file_path, n_mfcc=40, max_len=200):
#     try:
#         y, sr = librosa.load(file_path, sr=None)  # load file wav
#         mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
#         # mfcc = librosa.feature.mfcc(
#         #     y=y,
#         #     sr=sr,
#         #     n_mfcc=n_mfcc,
#         #     n_fft=512,        # cửa sổ FFT nhỏ hơn để tăng số frame
#         #     hop_length=256    # bước nhảy nhỏ để nhiều frame hơn
#         # )
#         # Padding hoặc cắt cho cùng độ dài
#         if mfcc.shape[1] < max_len:
#             pad_width = max_len - mfcc.shape[1]
#             mfcc = np.pad(mfcc, pad_width=((0,0),(0,pad_width)), mode='constant')
#         else:
#             mfcc = mfcc[:, :max_len]
#         return mfcc
#     except Exception as e:
#         print(f"❌ Error processing {file_path}: {e}")
#         return None


# def load_dataset(data_folder):
#     mfccs = []
#     labels = []
#     for idx, class_name in enumerate(sorted(os.listdir(data_folder))):
#         class_path = os.path.join(data_folder, class_name)
#         if not os.path.isdir(class_path):
#             continue
#         for file_name in os.listdir(class_path):
#             if file_name.lower().endswith(".wav"):
#                 file_path = os.path.join(class_path, file_name)
#                 features = extract_features(file_path)
#                 if features is not None:
#                     mfccs.append(features)
#                     labels.append(idx)

#     return np.array(mfccs), np.array(labels)