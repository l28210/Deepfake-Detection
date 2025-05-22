import socket
import struct
import threading
import json
import base64
import io
from PIL import Image
import torch
import torchvision.transforms as transforms
from model_load import model_load
import numpy as np

# Configuration\
HOST = ''           # Listen on all interfaces
PORT = 12345        # Match Qt client port
MODEL_PATH = r"F:\Software_Engineering\Deepfake-Detection-main\central_seed114514\19.pth"
DEVICE = torch.device('cpu')

# Load AI model
model = model_load(MODEL_PATH, DEVICE)
model.eval()

# Preprocessing pipeline
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# RPC method registry for extensibility
def ai_identify(params):
    # 解码并推理
    img_b64 = params.get('image')
    data = base64.b64decode(img_b64)
    image = Image.open(io.BytesIO(data)).convert('RGB')
    x = preprocess(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)[0]
    probs = torch.nn.functional.softmax(logits, dim=0)

    # 构造与原来完全相同的文本
    lines = ["=== Raw logits (y) ==="]
    for i, v in enumerate(logits):
        lines.append(f"y[{i}] = {v.item():.4f}")
    lines.append("")  # 空行
    lines.append("=== Softmax probabilities ===")
    for i, p in enumerate(probs):
        lines.append(f"Class {i}: {p.item():.4f}")

    result_str = "\n".join(lines)
    # JSON-RPC 2.0 里把它放在 result 里
    return result_str


# Placeholder for other RPC methods
def ai_generate(params):
    # params: {'image1': <b64>, 'image2': <b64>}
    # ... implement AI generation logic, return b64 of generated image
    return {'image': '<b64_generated_image>'}

def generate_report(params):
    # params: arbitrary report parameters
    # ... implement report logic
    return {'report': 'Report content here'}

RPC_METHODS = {
    'ai_identify': ai_identify,
    'ai_generate': ai_generate,
    'generate_report': generate_report,
}

class ClientHandler(threading.Thread):
    def __init__(self, conn, addr):
        super().__init__(daemon=True)
        self.conn = conn
        self.addr = addr

    def run(self):
        print(f"[+] Connection from {self.addr}")
        try:
            while True:
                # Read length header
                raw_len = self.conn.recv(4)
                if not raw_len:
                    break
                msg_len = struct.unpack('!I', raw_len)[0]
                if msg_len <= 0 or msg_len > 100_000_000:
                    break

                # Read JSON payload
                data = b''
                while len(data) < msg_len:
                    packet = self.conn.recv(msg_len - len(data))
                    if not packet:
                        break
                    data += packet
                if len(data) < msg_len:
                    break

                # Parse JSON-RPC request
                req = json.loads(data.decode('utf-8'))
                method = req.get('method')
                params = req.get('params', {})
                req_id = req.get('id')

                # Dispatch
                if method in RPC_METHODS:
                    try:
                        result = RPC_METHODS[method](params)
                        response = {'jsonrpc': '2.0', 'result': result, 'id': req_id}
                    except Exception as e:
                        response = {'jsonrpc': '2.0', 'error': {'code': -32000, 'message': str(e)}, 'id': req_id}
                else:
                    response = {'jsonrpc': '2.0', 'error': {'code': -32601, 'message': 'Method not found'}, 'id': req_id}

                resp_bytes = json.dumps(response).encode('utf-8')
                # Send length + payload
                self.conn.sendall(struct.pack('!I', len(resp_bytes)) + resp_bytes)

        except Exception as e:
            print(f"[!] Handler error: {e}")
        finally:
            self.conn.close()
            print(f"[-] Disconnected {self.addr}")

class RPCServer(threading.Thread):
    def __init__(self, host, port):
        super().__init__(daemon=True)
        self.host = host
        self.port = port

    def run(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((self.host, self.port))
            s.listen()
            print(f"[+] RPC Server listening on {self.host}:{self.port}")
            while True:
                conn, addr = s.accept()
                handler = ClientHandler(conn, addr)
                handler.start()

class ControlThread(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)

    def run(self):
        # Main control loop
        while True:
            # Insert periodic tasks, health checks, logging, etc.
            pass

if __name__ == '__main__':
    control = ControlThread()
    rpc_server = RPCServer(HOST, PORT)
    control.start()
    rpc_server.start()
    control.join()  # Keep main alive
