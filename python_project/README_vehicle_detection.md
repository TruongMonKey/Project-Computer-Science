# Hệ thống nhận diện và đếm phương tiện giao thông

## Mô tả
Hệ thống sử dụng AI (YOLOv8) để tự động nhận diện, phân loại và đếm các loại phương tiện giao thông từ video hoặc camera thời gian thực.

## Tính năng chính

### 🚗 Nhận diện phương tiện
- **Ô tô** (Car)
- **Xe tải** (Truck) 
- **Xe buýt** (Bus)
- **Xe máy** (Motorcycle)
- **Xe đạp** (Bicycle)
- **Người đi bộ** (Person)

### 📊 Thống kê và báo cáo
- Đếm phương tiện theo thời gian thực
- Thống kê chi tiết theo ngày/tuần/tháng
- Biểu đồ phân tích xu hướng
- Xuất báo cáo CSV/Excel
- Lưu trữ dữ liệu vào database

### 🎯 Tính năng nổi bật
- **Xử lý thời gian thực**: FPS cao, độ trễ thấp
- **Giao diện web**: Responsive, dễ sử dụng
- **Cấu hình linh hoạt**: Tùy chỉnh đường đếm
- **Upload video**: Hỗ trợ nhiều định dạng
- **Đa luồng**: Xử lý nhiều camera đồng thời

## Cài đặt

### Yêu cầu hệ thống
- Python 3.8+
- MySQL 8.0+
- RAM: 4GB+ (8GB+ khuyến nghị)
- GPU: NVIDIA (tùy chọn, để tăng tốc)

### Bước 1: Clone repository
```bash
git clone <repository-url>
cd YOLOv8-Traffic-Monitor/python_project
```

### Bước 2: Tạo virtual environment
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc
venv\Scripts\activate     # Windows
```

### Bước 3: Cài đặt dependencies
```bash
pip install -r requirements_vehicle.txt
```

### Bước 4: Cấu hình database
1. Tạo database MySQL:
```sql
CREATE DATABASE datn;
USE datn;

CREATE TABLE vehicle_statistics (
    id INT AUTO_INCREMENT PRIMARY KEY,
    date_recorded DATE NOT NULL,
    car_count INT DEFAULT 0,
    truck_count INT DEFAULT 0,
    bus_count INT DEFAULT 0,
    motorcycle_count INT DEFAULT 0,
    bicycle_count INT DEFAULT 0,
    total_count INT DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

2. Cập nhật thông tin database trong `app_vehicle_detection.py`:
```python
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'your_username'
app.config['MYSQL_PASSWORD'] = 'your_password'
app.config['MYSQL_DB'] = 'datn'
```

### Bước 5: Tải model YOLO
```bash
# Tạo thư mục cho model
mkdir -p YoloWeights

# Tải model (nếu chưa có)
# Có thể sử dụng model có sẵn hoặc train custom model
```

## Sử dụng

### Khởi động hệ thống
```bash
python app_vehicle_detection.py
```

Truy cập: http://localhost:5001

### Cách sử dụng

1. **Trang chủ**: Xem tổng quan hệ thống
2. **Nhận diện**: 
   - Chọn video hoặc upload video mới
   - Cấu hình đường đếm
   - Bắt đầu nhận diện
   - Xem kết quả thời gian thực
3. **Thống kê**:
   - Xem biểu đồ phân tích
   - Lọc theo khoảng thời gian
   - Xuất báo cáo
4. **Quy định**:
   - Xem các quy định giao thông
   - Thông tin về luật giao thông đường bộ
   - Bảng tóm tắt các hành vi bị nghiêm cấm

### API Endpoints

#### Nhận diện
- `POST /api/start_detection` - Bắt đầu nhận diện
- `POST /api/stop_detection` - Dừng nhận diện
- `GET /api/get_statistics` - Lấy thống kê thời gian thực

#### Quản lý video
- `GET /api/get_video_list` - Lấy danh sách video
- `POST /api/upload_video` - Upload video mới

#### Thống kê
- `GET /api/get_daily_statistics` - Lấy thống kê theo ngày
- `POST /api/save_statistics` - Lưu thống kê

## Cấu trúc project

```
python_project/
├── app_vehicle_detection.py      # Flask app chính
├── vehicle_detection.py          # Core detection logic
├── templates/
│   ├── vehicle_index.html        # Trang chủ
│   ├── vehicle_detection.html    # Trang nhận diện
│   └── vehicle_statistics.html   # Trang thống kê
├── static/                       # CSS, JS, images
├── Videos/                       # Thư mục chứa video
├── YoloWeights/                  # Model weights
├── Data/                         # Kết quả và dữ liệu
└── requirements_vehicle.txt      # Dependencies
```

## Tùy chỉnh

### Thay đổi model
```python
# Trong vehicle_detection.py
class VehicleDetectionSystem:
    def __init__(self, model_path='YoloWeights/best.pt'):
        self.model = YOLO(model_path)
```

### Cấu hình đường đếm
```python
# Tọa độ đường đếm (x1, y1, x2, y2)
line_start = (337, 391)  # Điểm bắt đầu
line_end = (917, 387)    # Điểm kết thúc
```

### Thêm loại phương tiện mới
```python
# Trong vehicle_detection.py
def get_vehicle_class_name(self, class_id):
    class_mapping = {
        0: 'person',
        1: 'bicycle', 
        2: 'car',
        3: 'motorcycle',
        5: 'bus',
        7: 'truck',
        # Thêm class mới ở đây
    }
```

## Troubleshooting

### Lỗi thường gặp

1. **ImportError: No module named 'cv2'**
   ```bash
   pip install opencv-python
   ```

2. **MySQL connection error**
   - Kiểm tra thông tin database
   - Đảm bảo MySQL service đang chạy

3. **Model not found**
   - Kiểm tra đường dẫn model trong `YoloWeights/`
   - Tải lại model nếu cần

4. **Memory error**
   - Giảm batch size
   - Sử dụng GPU nếu có
   - Tăng RAM

### Performance optimization

1. **Tăng tốc GPU**:
   ```python
   model = YOLO('model.pt')
   model.to('cuda')  # Sử dụng GPU
   ```

2. **Giảm độ phân giải**:
   ```python
   results = model(frame, imgsz=320)  # Giảm kích thước
   ```

3. **Batch processing**:
   ```python
   results = model(frames, batch=4)  # Xử lý batch
   ```

## Đóng góp

1. Fork project
2. Tạo feature branch
3. Commit changes
4. Push to branch
5. Tạo Pull Request

## License

MIT License - xem file LICENSE để biết thêm chi tiết.

## Liên hệ

- Email: your.email@example.com
- GitHub: https://github.com/your-username
- Project: https://github.com/your-username/vehicle-detection-system
