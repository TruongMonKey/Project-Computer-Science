import cv2
import torch

class VehicleDetector:
    """
    Hệ thống nhận diện phương tiện giao thông sử dụng YOLOv8.
    Chỉ nhận diện và vẽ bounding box, không đếm người.
    """

    def __init__(self, model_path="yolov8s.pt", device='cpu'):
        """
        Khởi tạo mô hình YOLOv8.
        """
        self.device = device
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True).to(device)
        self.classes_of_interest = ['car', 'truck', 'bus', 'motorcycle', 'bicycle']  # Lọc các class phương tiện

    def process_frame(self, frame):
        """
        Nhận diện phương tiện trên 1 frame và vẽ bounding box.
        """
        # Chạy mô hình YOLO
        results = self.model(frame)

        # Lấy kết quả dạng dict
        detections = results.pandas().xyxy[0]  # x_min, y_min, x_max, y_max, confidence, class, name

        for _, row in detections.iterrows():
            cls_name = row['name']
            if cls_name in self.classes_of_interest:
                x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
                confidence = row['confidence']

                # Vẽ bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{cls_name} {confidence:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return frame

    def predict(self, frame):
        """
        Trả về kết quả detection (không vẽ), dùng cho mục đích khác nếu cần.
        """
        results = self.model(frame)
        detections = results.pandas().xyxy[0]
        return detections[detections['name'].isin(self.classes_of_interest)]
