import json
import socket
import threading


class EnvSocketServer:
    """
    简易的socket服务器，用于和前端通信。
    启动后等待前端连接，接收偏好更新消息，并支持发送位置信息给前端。
    """

    def __init__(self, host='127.0.0.1', port=9999):
        self.host = host
        self.port = port
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind((self.host, self.port))
        self.server.listen(1)
        self.conn = None
        self.addr = None
        self.env = None  # 后续会在HighCommandEnv中设置

        # 启动独立线程监听连接
        threading.Thread(target=self.accept_connection, daemon=True).start()

    def accept_connection(self):
        print("等待前端连接...")
        try:
            self.conn, self.addr = self.server.accept()
            print(f"前端已连接：{self.addr}")
            self.listen_for_messages()
        except Exception as e:
            print(f"接受连接时发生异常：{e}")

    def listen_for_messages(self):
        buffer = ""
        while True:
            try:
                data = self.conn.recv(4096)
                if not data:
                    print("前端已断开连接")
                    break
                buffer += data.decode('utf-8')
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    line = line.strip()
                    if not line:
                        continue
                    self.on_message(line)
            except Exception as e:
                print(f"接收消息时发生异常：{e}")
                break
        self.conn.close()
        self.conn = None
        self.addr = None

    def on_message(self, msg):
        try:
            info = json.loads(msg)
            if info.get("type") == "update_preference":
                # 接收前端传来的偏好数据并更新env
                if self.env is not None:
                    # 假设data是一个二维列表，如 [[0.5, 0.7, 0.3], [0.2, 0.0, 0.9]]
                    preference_data = info.get("data", [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
                    # 验证数据格式和范围
                    if (isinstance(preference_data, list) and
                        len(preference_data) == 2 and
                        all(isinstance(sub, list) and len(sub) == 3 for sub in preference_data)):
                        # 确保所有值在0.0到1.0之间
                        valid = all(0.0 <= val <= 1.0 for sub in preference_data for val in sub)
                        if valid:
                            self.env.set_human_preference(preference_data)
                            print(f"接收到新的偏好数据：{preference_data}")
                        else:
                            print("偏好值超出范围（0.0到1.0）。")
                    else:
                        print("偏好数据格式错误。")
        except json.JSONDecodeError as e:
            print(f"JSON解析失败：{e}")

    def send(self, data_str):
        if self.conn:
            try:
                self.conn.sendall(data_str.encode('utf-8'))
            except Exception as e:
                print(f"发送数据时发生异常：{e}")
                self.conn.close()
                self.conn = None
                self.addr = None