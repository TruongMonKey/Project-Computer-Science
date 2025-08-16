#!/usr/bin/env python3
"""
Script khởi động hệ thống nhận diện và đếm phương tiện giao thông
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def check_dependencies():
    """Kiểm tra các dependencies cần thiết"""
    print("🔍 Kiểm tra dependencies...")
    
    required_packages = [
        'flask',
        'ultralytics', 
        'opencv-python',
        'supervision',
        'numpy',
        'flask-mysqldb'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} - CHƯA CÀI ĐẶT")
    
    if missing_packages:
        print(f"\n⚠️  Thiếu {len(missing_packages)} package:")
        for pkg in missing_packages:
            print(f"   - {pkg}")
        
        install = input("\n❓ Bạn có muốn cài đặt các package thiếu? (y/N): ")
        if install.lower() in ['y', 'yes']:
            install_packages(missing_packages)
        else:
            print("❌ Không thể chạy hệ thống thiếu dependencies!")
            return False
    
    return True

def install_packages(packages):
    """Cài đặt các package thiếu"""
    print("\n📦 Đang cài đặt packages...")
    
    for package in packages:
        try:
            print(f"   Đang cài {package}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f"   ✅ Đã cài {package}")
        except subprocess.CalledProcessError:
            print(f"   ❌ Lỗi khi cài {package}")
            return False
    
    print("✅ Cài đặt hoàn tất!")
    return True

def check_files():
    """Kiểm tra các file cần thiết"""
    print("\n📁 Kiểm tra cấu trúc project...")
    
    required_files = [
        'app_vehicle_detection.py',
        'vehicle_detection.py',
        'templates/vehicle_index.html',
        'templates/vehicle_detection.html', 
        'templates/vehicle_statistics.html',
        'requirements_vehicle.txt'
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            missing_files.append(file_path)
            print(f"❌ {file_path} - KHÔNG TỒN TẠI")
    
    if missing_files:
        print(f"\n⚠️  Thiếu {len(missing_files)} file:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    
    return True

def check_directories():
    """Tạo các thư mục cần thiết"""
    print("\n📂 Kiểm tra thư mục...")
    
    required_dirs = [
        'Videos',
        'Data/Example Results',
        'YoloWeights',
        'static',
        'templates'
    ]
    
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
            print(f"📁 Đã tạo: {dir_path}")
        else:
            print(f"✅ {dir_path}")

def check_model():
    """Kiểm tra model YOLO"""
    print("\n🤖 Kiểm tra model YOLO...")
    
    model_paths = [
        'YoloWeights/best.pt',
        'YoloWeights/yolov8n.pt',
        'YoloWeights/yolov8s.pt'
    ]
    
    found_models = []
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            size = os.path.getsize(model_path) / (1024 * 1024)  # MB
            print(f"✅ {model_path} ({size:.1f} MB)")
            found_models.append(model_path)
        else:
            print(f"❌ {model_path} - KHÔNG TỒN TẠI")
    
    if not found_models:
        print("\n⚠️  Không tìm thấy model nào!")
        print("💡 Bạn có thể:")
        print("   1. Tải model từ https://github.com/ultralytics/yolov8")
        print("   2. Train custom model")
        print("   3. Sử dụng model mặc định (sẽ tự động tải)")
        
        continue_without_model = input("\n❓ Tiếp tục mà không có model? (y/N): ")
        if continue_without_model.lower() not in ['y', 'yes']:
            return False
    
    return True

def start_server():
    """Khởi động Flask server"""
    print("\n🚀 Khởi động hệ thống...")
    
    try:
        # Import và chạy app
        from app_vehicle_detection import app
        
        print("✅ Server đã sẵn sàng!")
        print("\n" + "="*50)
        print("🎉 HỆ THỐNG NHẬN DIỆN PHƯƠNG TIỆN")
        print("="*50)
        print("📍 URL: http://localhost:5001")
        print("📱 Giao diện: http://localhost:5001/detection")
        print("📊 Thống kê: http://localhost:5001/statistics")
        print("📖 Quy định: http://localhost:5001/regulations")
        print("\n💡 Nhấn Ctrl+C để dừng server")
        print("="*50)
        
        # Chạy app
        app.run(debug=True, host='0.0.0.0', port=5001)
        
    except ImportError as e:
        print(f"❌ Lỗi import: {e}")
        return False
    except Exception as e:
        print(f"❌ Lỗi khởi động server: {e}")
        return False

def main():
    """Hàm main"""
    print("🚗 HỆ THỐNG NHẬN DIỆN VÀ ĐẾM PHƯƠNG TIỆN GIAO THÔNG")
    print("="*60)
    
    # Kiểm tra Python version
    if sys.version_info < (3, 8):
        print("❌ Yêu cầu Python 3.8+")
        return
    
    print(f"✅ Python {sys.version}")
    
    # Kiểm tra dependencies
    if not check_dependencies():
        return
    
    # Kiểm tra files
    if not check_files():
        print("\n❌ Thiếu file cần thiết!")
        print("💡 Chạy script cleanup_project.py trước")
        return
    
    # Kiểm tra thư mục
    check_directories()
    
    # Kiểm tra model
    if not check_model():
        return
    
    # Khởi động server
    start_server()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Đã dừng server!")
    except Exception as e:
        print(f"\n❌ Lỗi: {e}")
        print("💡 Kiểm tra log để biết thêm chi tiết")
