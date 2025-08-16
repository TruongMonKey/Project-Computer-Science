import os
from typing import Dict, Tuple, Optional

import cv2
import numpy as np
from ultralytics import YOLO
from sort import Sort  # cần có sort.py cùng thư mục, hoặc `pip install sort-tracker`


class VehicleDetectionSystem:
    """
    Hệ thống nhận diện + tracking + đếm phương tiện qua line sử dụng YOLOv8 + SORT.
    - Map class theo COCO: {1: bicycle, 2: car, 3: motorcycle, 5: bus, 7: truck}
    - Gán class cho track bằng IoU giữa bbox track và bbox detect của frame hiện tại.
    - Đếm khi track đi từ 1 phía của line sang phía còn lại (tránh đếm trùng).
    """

    # COCO vehicle classes
    VEHICLE_CLASS_IDS = {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

    def __init__(
        self,
        yolo_weights: str = "YoloWeights/yolov8s.pt",
        conf_thres: float = 0.3,
    ):
        # Load YOLOv8 (nếu dùng model custom, giữ đúng đường dẫn)
        self.model = YOLO(yolo_weights)

        # SORT tracker
        self.tracker = Sort()  # có thể truyền max_age, min_hits, iou_threshold nếu cần

        # Tham số
        self.conf_thres = conf_thres

        # Line để đếm: ((x1, y1), (x2, y2))
        self.counting_line: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None

        # Thống kê
        self.counts: Dict[str, int] = {
            "car": 0,
            "truck": 0,
            "bus": 0,
            "motorcycle": 0,
            "bicycle": 0,
            "total": 0,
        }

        # Track đã đếm
        self.tracked_ids = set()

        # Ghi nhớ class cho mỗi track_id
        self.track_classes: Dict[int, str] = {}

        # Ghi nhớ phía (sign) của tâm track so với line để phát hiện crossing
        self.track_last_side: Dict[int, int] = {}

    # ---------- Public API cho Flask ----------

    def setup_counting_line(self, start: Tuple[int, int], end: Tuple[int, int]):
        """Thiết lập line đếm."""
        self.counting_line = (start, end)

    def reset_counts(self):
        """Reset thống kê và trạng thái tracking (dùng khi bắt đầu video mới)."""
        for k in self.counts:
            self.counts[k] = 0
        self.tracked_ids.clear()
        self.track_classes.clear()
        self.track_last_side.clear()

    def process_frame(self, frame):
        """
        Xử lý 1 frame:
        - YOLO detect (lọc class phương tiện)
        - SORT track
        - Gán class cho track bằng IoU với detections
        - Kiểm tra crossing line để đếm
        - Vẽ kết quả lên frame
        """
        if frame is None or frame.size == 0:
            return frame

        # 1) YOLO detect (lọc theo class và conf)
        # Lưu ý: 'classes' chỉ áp dụng nếu model là COCO. Nếu dùng model custom, bỏ `classes=...`
        results = self.model(
            frame,
            verbose=False,
            conf=self.conf_thres,
            classes=list(self.VEHICLE_CLASS_IDS.keys()),
        )[0]

        det_boxes = []   # [[x1,y1,x2,y2]]
        det_scores = []  # [conf]
        det_clsids = []  # [int]

        if results.boxes is not None and len(results.boxes) > 0:
            for b in results.boxes:
                # xyxy có shape (1,4) -> lấy hàng đầu
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                conf = float(b.conf[0].item())
                clsid = int(b.cls[0].item())
                # đảm bảo toạ độ int và trong ảnh
                det_boxes.append([int(x1), int(y1), int(x2), int(y2)])
                det_scores.append(conf)
                det_clsids.append(clsid)

        # 2) SORT update: đầu vào dạng [x1, y1, x2, y2, score]
        dets_for_sort = []
        for box, score in zip(det_boxes, det_scores):
            dets_for_sort.append([box[0], box[1], box[2], box[3], score])
        dets_for_sort = np.array(dets_for_sort, dtype=float) if dets_for_sort else np.empty((0, 5))

        tracked_objects = self.tracker.update(dets_for_sort)  # Nx5: x1,y1,x2,y2,track_id

        # 3) Gán class cho từng track qua IoU với det boxes
        #    tracked_objects shape (N,5). Nếu N==0 thì bỏ qua
        if len(tracked_objects) > 0 and len(det_boxes) > 0:
            track_boxes = tracked_objects[:, :4]
            matches = self._match_tracks_to_dets_iou(track_boxes, det_boxes, iou_thres=0.1)
            # matches: list of (t_idx, d_idx)
            for t_idx, d_idx in matches:
                track_id = int(tracked_objects[t_idx, 4])
                clsid = det_clsids[d_idx]
                cls_name = self.VEHICLE_CLASS_IDS.get(clsid)
                if cls_name:
                    self.track_classes[track_id] = cls_name

        # 4) Vẽ line
        if self.counting_line:
            (lx1, ly1), (lx2, ly2) = self.counting_line
            cv2.line(frame, (lx1, ly1), (lx2, ly2), (0, 0, 255), 2)

        # 5) Xử lý từng track: đếm crossing + vẽ box/label
        for trk in tracked_objects:
            x1, y1, x2, y2, tid = trk
            x1, y1, x2, y2, tid = int(x1), int(y1), int(x2), int(y2), int(tid)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            cls_name = self.track_classes.get(tid, None)

            # Đếm crossing nếu có line & biết class
            if self.counting_line and cls_name is not None:
                side = self._point_side_of_line((cx, cy), self.counting_line)
                last_side = self.track_last_side.get(tid, None)

                if last_side is None:
                    # Khởi tạo phía ban đầu
                    self.track_last_side[tid] = side
                else:
                    # Khi đổi phía (cross), và chưa đếm cho tid này trước đó → đếm
                    if side != 0 and last_side != 0 and side != last_side and tid not in self.tracked_ids:
                        self.tracked_ids.add(tid)
                        self.counts[cls_name] += 1
                        self.counts["total"] += 1
                    # Cập nhật phía hiện tại
                    self.track_last_side[tid] = side

            # Vẽ box + label
            color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{cls_name or 'obj'}-{tid}"
            cv2.putText(frame, label, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.circle(frame, (cx, cy), 3, (255, 0, 0), -1)

        return frame

    def get_current_counts(self) -> Dict[str, int]:
        """Trả về dict thống kê hiện tại."""
        return dict(self.counts)

    def save_counts_to_file(self, filename: str) -> str:
        """Lưu thống kê vào file txt và trả về đường dẫn."""
        os.makedirs("Data/Example Results", exist_ok=True)
        filepath = os.path.join("Data/Example Results", filename)
        with open(filepath, "w", encoding="utf-8") as f:
            for k, v in self.counts.items():
                f.write(f"{k}: {v}\n")
        return filepath

    # ---------- Helpers ----------

    @staticmethod
    def _iou(boxA, boxB) -> float:
        """
        IoU của 2 box [x1,y1,x2,y2]
        """
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interW = max(0, xB - xA + 1)
        interH = max(0, yB - yA + 1)
        interArea = interW * interH

        if interArea <= 0:
            return 0.0

        boxAArea = max(0, (boxA[2] - boxA[0] + 1)) * max(0, (boxA[3] - boxA[1] + 1))
        boxBArea = max(0, (boxB[2] - boxB[0] + 1)) * max(0, (boxB[3] - boxB[1] + 1))
        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-9)
        return float(iou)

    def _match_tracks_to_dets_iou(self, track_boxes, det_boxes, iou_thres=0.1):
        """
        Ghép mỗi track box với det box có IoU cao nhất (>= iou_thres).
        Trả về list các cặp (track_idx, det_idx).
        """
        matches = []
        if len(track_boxes) == 0 or len(det_boxes) == 0:
            return matches

        T = len(track_boxes)
        D = len(det_boxes)

        used_dets = set()
        for t_idx in range(T):
            best_iou = 0.0
            best_d = -1
            for d_idx in range(D):
                if d_idx in used_dets:
                    continue
                iou = self._iou(track_boxes[t_idx], det_boxes[d_idx])
                if iou > best_iou:
                    best_iou = iou
                    best_d = d_idx
            if best_d >= 0 and best_iou >= iou_thres:
                matches.append((t_idx, best_d))
                used_dets.add(best_d)
        return matches

    @staticmethod
    def _point_side_of_line(p: Tuple[int, int],
                            line: Tuple[Tuple[int, int], Tuple[int, int]]) -> int:
        """
        Xác định phía của điểm p so với line.
        Trả về:
          1  : một phía
         -1  : phía còn lại
          0  : nằm gần như trên line
        """
        (x1, y1), (x2, y2) = line
        x, y = p
        # Cross product sign
        val = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
        if val > 0:
            return 1
        elif val < 0:
            return -1
        else:
            return 0
