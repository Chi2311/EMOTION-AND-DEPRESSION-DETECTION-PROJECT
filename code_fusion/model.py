from torch import nn
from common import OCBAM

class Dual(nn.Module):
    """
    Kiến trúc CNN-2D kết hợp CBAM đã được tối ưu để chống Overfitting.
    Giảm số kênh ở các lớp sâu và tăng Dropout để tăng tính tổng quát.
    """
    def __init__(self, num_classes=3): 
        super(Dual, self).__init__()

        # --- Khối trích xuất đặc trưng CNN (Đã giảm độ phức tạp) ---
        self.feature_extractor = nn.Sequential(
            # Lớp 1
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Lớp 2
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Lớp 3
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(4),
            
            # Lớp 4
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # --- Cơ chế Attention (Điều chỉnh theo 128 kênh) ---
        self.cbam = OCBAM(128)

        # --- Khối phân loại (Tăng Dropout) ---
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.5), # Giảm nhẹ Dropout từ 0.6 xuống 0.5
            nn.Linear(128, 64),
            nn.BatchNorm1d(64), # THÊM VÀO ĐÂY: giúp ổn định phân phối nhãn
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, mfcc):
        features = self.feature_extractor(mfcc)
        refined_features = self.cbam(features)
        output = self.classifier(refined_features)
        return output