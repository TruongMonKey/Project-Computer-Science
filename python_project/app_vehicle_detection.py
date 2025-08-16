import datetime
import os
import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify, request, send_from_directory
from flask_cors import CORS
from flask_mysqldb import MySQL
import threading
import time
import base64

from python_project.vehicle_detections_system import VehicleDetectionSystem

app = Flask(__name__, static_folder='static')
CORS(app)

# Cấu hình MySQL
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'Truongkhi19'
app.config['MYSQL_DB'] = 'datn'
mysql = MySQL(app)

# Khởi tạo hệ thống detection
vehicle_detector = VehicleDetectionSystem()
current_video_path = None
is_processing = False
processing_thread = None

# Biến lưu trữ thống kê
vehicle_statistics = {
    'car': 0,
    'truck': 0,
    'bus': 0,
    'motorcycle': 0,
    'bicycle': 0,
    'total': 0,
    'last_update': datetime.datetime.now()
}

@app.route('/')
def index():
    """Trang chủ"""
    return render_template('vehicle_index.html')

@app.route('/detection')
def detection():
    """Trang nhận diện phương tiện"""
    return render_template('vehicle_detection.html')

@app.route('/statistics')
def statistics():
    """Trang thống kê"""
    return render_template('vehicle_statistics.html')

@app.route('/regulations')
def regulations():
    """Trang quy định giao thông"""
    return render_template('bb.html')

@app.route('/api/start_detection', methods=['POST'])
def start_detection():
    """API bắt đầu nhận diện"""
    global is_processing, current_video_path, processing_thread
    
    if is_processing:
        return jsonify({'status': 'error', 'message': 'Đang xử lý video khác'})
    
    data = request.get_json()
    video_path = data.get('video_path', 'Videos/test4.mp4')
    line_start = data.get('line_start', [337, 391])
    line_end = data.get('line_end', [917, 387])
    
    current_video_path = video_path

    
    # Thiết lập đường đếm
    vehicle_detector.setup_counting_line(line_start, line_end)
    
    # Bắt đầu xử lý trong thread riêng
    is_processing = True
    processing_thread = threading.Thread(
        target=process_video_thread,
        args=(video_path, line_start, line_end)
    )
    processing_thread.start()
    
    return jsonify({'status': 'success', 'message': 'Bắt đầu nhận diện'})

@app.route('/api/stop_detection', methods=['POST'])
def stop_detection():
    """API dừng nhận diện"""
    global is_processing
    
    is_processing = False
    return jsonify({'status': 'success', 'message': 'Đã dừng nhận diện'})

@app.route('/api/get_statistics')
def get_statistics():
    """API lấy thống kê"""
    global vehicle_statistics
    
    # Cập nhật thống kê từ detector
    current_counts = vehicle_detector.get_current_counts()
    vehicle_statistics.update(current_counts)
    vehicle_statistics['last_update'] = datetime.datetime.now()
    
    return jsonify(vehicle_statistics)

@app.route('/api/save_statistics', methods=['POST'])
def save_statistics():
    """API lưu thống kê"""
    try:
        # Lưu vào file
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"vehicle_statistics_{timestamp}.txt"
        vehicle_detector.save_counts_to_file(filename)
        
        # Lưu vào database
        save_to_database()
        
        return jsonify({'status': 'success', 'message': f'Đã lưu thống kê vào {filename}'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

def save_to_database():
    """Lưu thống kê vào database"""
    try:
        cur = mysql.connection.cursor()
        current_time = datetime.datetime.now()
        
        # Lưu thống kê theo ngày
        cur.execute("""
            INSERT INTO vehicle_statistics 
            (date_recorded, car_count, truck_count, bus_count, motorcycle_count, bicycle_count, total_count)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (
            current_time.date(),
            vehicle_statistics['car'],
            vehicle_statistics['truck'],
            vehicle_statistics['bus'],
            vehicle_statistics['motorcycle'],
            vehicle_statistics['bicycle'],
            vehicle_statistics['total']
        ))
        
        mysql.connection.commit()
        cur.close()
        
    except Exception as e:
        print(f"Lỗi khi lưu vào database: {e}")

@app.route('/api/get_daily_statistics')
def get_daily_statistics():
    """API lấy thống kê theo ngày"""
    try:
        cur = mysql.connection.cursor()
        cur.execute("""
            SELECT date_recorded, car_count, truck_count, bus_count, motorcycle_count, bicycle_count, total_count
            FROM vehicle_statistics
            ORDER BY date_recorded DESC
            LIMIT 30
        """)
        
        results = cur.fetchall()
        cur.close()
        
        statistics = []
        for row in results:
            statistics.append({
                'date': row[0].strftime('%Y-%m-%d'),
                'car': row[1],
                'truck': row[2],
                'bus': row[3],
                'motorcycle': row[4],
                'bicycle': row[5],
                'total': row[6]
            })
            
        return jsonify(statistics)
        
    except Exception as e:
        return jsonify({'error': str(e)})

def process_video_thread(video_path, line_start, line_end):
    """Xử lý video trong thread riêng"""
    global is_processing, vehicle_statistics
    
    try:
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Không thể mở video: {video_path}")
            is_processing = False
            return
            
        while is_processing:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Xử lý frame
            processed_frame = vehicle_detector.process_frame(frame)
            
            # Cập nhật thống kê
            current_counts = vehicle_detector.get_current_counts()
            vehicle_statistics.update(current_counts)
            
            # Delay để giảm tải CPU
            time.sleep(0.03)  # ~30 FPS
            
        cap.release()
        
    except Exception as e:
        print(f"Lỗi khi xử lý video: {e}")
    finally:
        is_processing = False

@app.route('/api/upload_video', methods=['POST'])
def upload_video():
    """API upload video"""
    try:
        if 'video' not in request.files:
            return jsonify({'status': 'error', 'message': 'Không có file video'})
            
        video_file = request.files['video']
        if video_file.filename == '':
            return jsonify({'status': 'error', 'message': 'Chưa chọn file'})
            
        # Lưu file
        filename = f"uploaded_{int(time.time())}.mp4"
        filepath = os.path.join('Videos', filename)
        video_file.save(filepath)
        
        return jsonify({
            'status': 'success', 
            'message': 'Upload thành công',
            'filepath': filepath
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/get_video_list')
def get_video_list():
    """API lấy danh sách video"""
    try:
        video_dir = 'Videos'
        video_files = []
        
        for filename in os.listdir(video_dir):
            if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                video_files.append(filename)
                
        return jsonify(video_files)
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/Videos/<path:filename>')
def serve_video(filename):
    return send_from_directory('Videos', filename)

if __name__ == '__main__':
    # Tạo thư mục nếu chưa có
    os.makedirs('Videos', exist_ok=True)
    os.makedirs('Data/Example Results', exist_ok=True)
    
    print("Khởi động hệ thống nhận diện phương tiện giao thông...")
    print("Truy cập: http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5001)
