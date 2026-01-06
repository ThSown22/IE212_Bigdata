import cv2
import socket
import base64
import json
import time
import numpy as np

# CẤU HÌNH STREAMING
class SenderConfig:
    # Host và Port để Receiver kết nối đến
    HOST = "localhost"      
    PORT = 6100             
    
    # Cấu hình Camera
    FRAME_WIDTH = 640       
    FRAME_HEIGHT = 480      
    
    # Cấu hình Streaming
    BATCH_INTERVAL = 2     
    JPEG_QUALITY = 80       

# HÀM KẾT NỐI TCP
def create_tcp_connection():
    """
    Khởi tạo kết nối TCP Server.
    Server sẽ chờ Receiver kết nối đến.
    
    Returns:
        connection: Đối tượng socket để gửi dữ liệu
    """
    # Lấy thông tin cấu hình
    TCP_IP = SenderConfig.HOST
    TCP_PORT = SenderConfig.PORT
    
    # Tạo socket TCP
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    # Bind và lắng nghe kết nối
    server_socket.bind((TCP_IP, TCP_PORT))
    server_socket.listen(1)
    
    print(f"[SENDER] Đang chờ kết nối tại {TCP_IP}:{TCP_PORT}...")
    
    # Chấp nhận kết nối từ Receiver
    connection, address = server_socket.accept()
    print(f"[SENDER] Đã kết nối với Receiver: {address}")
    
    return connection, server_socket

# HÀM CHUYỂN ĐỔI FRAME 
def frame_to_string(frame):
    """
    Chuyển đổi frame (ma trận numpy) thành chuỗi văn bản.
    
    Quy trình:
    1. Frame là ma trận 3 chiều
    2. Flatten thành ma trận 1 chiều
    3. Encode thành base64 string để truyền qua TCP
    
    Args:
        frame: numpy array của hình ảnh (BGR format từ OpenCV)
    
    Returns:
        str: Chuỗi base64 của frame đã encode
    """
    # Encode frame thành JPEG để giảm kích thước
    success, encoded_img = cv2.imencode(
        ".jpg", frame,
        [int(cv2.IMWRITE_JPEG_QUALITY), SenderConfig.JPEG_QUALITY]
    )
    
    if not success:
        return None
    
    # Chuyển bytes thành base64 string (văn bản)
    # => Điều này giúp truyền dữ liệu dễ dàng qua socket text streaming
    base64_string = base64.b64encode(encoded_img.tobytes()).decode("utf-8")
    
    return base64_string

def send_frame(connection, frame_data, frame_id):
    """
    Gửi một frame qua TCP connection.
    
    Args:
        connection: Socket connection
        frame_data: Dữ liệu frame đã encode thành string
        frame_id: ID của frame (để tracking)
    """
    # Tạo payload dạng dictionary
    payload = {
        "frame_id": frame_id,
        "data": frame_data,
        "timestamp": time.time(),
        "width": SenderConfig.FRAME_WIDTH,
        "height": SenderConfig.FRAME_HEIGHT
    }
    
    # Chuyển thành JSON string và thêm ký tự xuống dòng (signal) đánh dấu kết thúc một batch
    message = json.dumps(payload) + "\n"
    
    # Gửi qua TCP
    connection.sendall(message.encode("utf-8"))

# ==========================
# MAIN - STREAMING PROCESS
# ==========================
def main():
    """
    Hàm chính để chạy Sender module.
    
    Quy trình:
    1. Khởi tạo kết nối TCP (chờ Receiver kết nối)
    2. Mở camera
    3. Liên tục đọc frame từ camera
    4. Chuyển frame thành văn bản (base64)
    5. Gửi qua TCP đến Receiver
    """
    print("=" * 60)
    print("          SENDER MODULE - CAMERA SERVER")
    print("=" * 60)
    
    # Bước 1: Khởi tạo kết nối TCP
    print("\n[SENDER] Khởi tạo kết nối TCP...")
    tcp_connection, server_socket = create_tcp_connection()
    
    # Bước 2: Mở camera
    print("[SENDER] Đang mở camera...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("[ERROR] Không thể mở camera!")
        print("[INFO] Đang chuyển sang chế độ tạo frame giả lập...")
        use_fake_frames = True
    else:
        use_fake_frames = False
        # Cấu hình camera
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, SenderConfig.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, SenderConfig.FRAME_HEIGHT)
        print("[SENDER] Camera đã sẵn sàng!")
    
    print(f"\n[SENDER] Bắt đầu streaming với batch interval = {SenderConfig.BATCH_INTERVAL}s")
    print("[SENDER] Nhấn Ctrl+C để dừng...\n")
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            if use_fake_frames:
                # Tạo frame giả lập (gradient màu thay đổi theo thời gian)
                frame = np.zeros((SenderConfig.FRAME_HEIGHT, SenderConfig.FRAME_WIDTH, 3), dtype=np.uint8)
                color_shift = int((time.time() * 50) % 255)
                frame[:, :] = [color_shift, 100, 200 - color_shift // 2]
                # Thêm text hiển thị frame ID
                cv2.putText(frame, f"Frame #{frame_count}", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            else:
                # Đọc frame từ camera
                ret, frame = cap.read()
                
                if not ret:
                    print("[WARN] Không đọc được frame từ camera")
                    continue
                
                # Flip frame để không bị ngược (mirror effect)
                frame = cv2.flip(frame, 1)
                
                # Resize frame theo cấu hình
                frame = cv2.resize(frame, (SenderConfig.FRAME_WIDTH, SenderConfig.FRAME_HEIGHT))
            
            # Chuyển frame thành string (văn bản)
            frame_string = frame_to_string(frame)
            
            if frame_string is None:
                print("[WARN] Không thể encode frame")
                continue
            
            # Gửi frame qua TCP
            try:
                send_frame(tcp_connection, frame_string, frame_count)
                frame_count += 1
                
                # Hiển thị thông tin streaming
                elapsed = time.time() - start_time
                print(f"[SENDER] Đã gửi Frame #{frame_count} | "
                      f"Elapsed: {elapsed:.1f}s | "
                      f"Size: {len(frame_string)} bytes")
                
            except BrokenPipeError:
                print("[ERROR] Kết nối bị ngắt!")
                break
            except Exception as e:
                print(f"[ERROR] Lỗi khi gửi frame: {e}")
                break
            
            # Chờ theo batch interval
            time.sleep(SenderConfig.BATCH_INTERVAL)
    
    except KeyboardInterrupt:
        print("\n[SENDER] Đang dừng streaming...")
    
    finally:
        # Giải phóng tài nguyên
        if not use_fake_frames:
            cap.release()
        tcp_connection.close()
        server_socket.close()
        print("[SENDER] Đã giải phóng tài nguyên!")
        print(f"[SENDER] Tổng số frame đã gửi: {frame_count}")


if __name__ == "__main__":
    main()
