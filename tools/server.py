import os
import sys
import torch
import argparse
import threading
import queue
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import numpy as np
import base64
from PIL import Image
import io
import tempfile
import torchvision
import matplotlib.pyplot as plt
import glob
import shutil

# 设置工作目录和路径
# __file__ 在 tools/server.py，两个 dirname 刚好回到项目根
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

# 导入后端功能模块
from model_load import model_load
from run import run_face_swapping
from visualize_erf import parse_args, main as visualize_erf_main

# 创建Flask应用
app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 结果队列和处理状态字典
result_queue = queue.Queue()
processing_status = {}

# 模型加载和初始化
device = torch.device('cpu')  # 默认使用CPU
detection_model = None

# 初始化检测模型
def init_detection_model():
    global detection_model
    model_path = os.path.join(BASE_DIR, "central_seed114514", "19.pth")
    detection_model = model_load(model_path, device)
    detection_model.eval()
    print("检测模型初始化完成")

# 图像检测线程函数
def detection_thread(image_data, task_id):
    try:
        # 解码图像数据
        img = Image.open(io.BytesIO(base64.b64decode(image_data)))
        img = img.convert('RGB')
        
        # 图像预处理
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((256, 256)),
            torchvision.transforms.ToTensor()
        ])
        
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        # 模型推理
        with torch.no_grad():
            output = detection_model(img_tensor)
            print(output)
            fake_conf = output[0][0].item()
            real_conf = output[0][1].item()
            print(fake_conf > real_conf)
        
        # 返回结果
        result = {
            "task_id": task_id,
            "status": "success",
            "is_fake": fake_conf > real_conf,
            "fake_confidence": fake_conf,
            "real_confidence": real_conf
        }
        
        # DEBUG 打印一下
        import json
        print(">> 返回前端的 JSON:", json.dumps(result, ensure_ascii=False))

    except Exception as e:
        result = {
            "task_id": task_id,
            "status": "error",
            "message": str(e)
        }
    
    # 将结果放入队列
    result_queue.put(result)
    # 更新处理状态
    processing_status[task_id] = "completed"

# 换脸线程函数
def face_swapping_thread(source_image_data, target_image_data, task_id):
    try:
        # 创建临时文件
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as source_file, \
             tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as target_file, \
             tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as output_file:
            
            # 保存图像数据到临时文件
            source_file.write(base64.b64decode(source_image_data))
            target_file.write(base64.b64decode(target_image_data))
            
            # 执行换脸
            run_face_swapping(
                source_file.name, 
                target_file.name, 
                output_file.name
            )
            
            # 读取结果图像
            with open(output_file.name, 'rb') as f:
                result_image = base64.b64encode(f.read()).decode('utf-8')
        
        # 清理临时文件
        os.unlink(source_file.name)
        os.unlink(target_file.name)
        os.unlink(output_file.name)
        
        # 返回结果
        result = {
            "task_id": task_id,
            "status": "success",
            "result_image": result_image
        }
        
    except Exception as e:
        result = {
            "task_id": task_id,
            "status": "error",
            "message": str(e)
        }
    
    # 将结果放入队列
    result_queue.put(result)
    # 更新处理状态
    processing_status[task_id] = "completed"

# 热力图生成线程函数
def heatmap_generation_thread(model_file, image_data, task_id):
    try:
        # 保存模型文件到临时目录
        model_dir = os.path.join(tempfile.gettempdir(), f"model_{task_id}")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "model.pth")
        
        with open(model_path, 'wb') as f:
            f.write(base64.b64decode(model_file))
        
        # 保存图像到临时文件
        image_path = os.path.join(model_dir, "image.jpg")
        with open(image_path, 'wb') as f:
            f.write(base64.b64decode(image_data))
        
        # 设置参数
        args = parse_args()
        args.model_path = model_path
        args.data_path = os.path.dirname(image_path)
        args.save_path = os.path.join(model_dir, "erf.npy")
        
        # 执行热力图生成
        visualize_erf_main(args)
        
        # 生成可视化图像
        import matplotlib.pyplot as plt
        erf_matrix = np.load(args.save_path)
        
        plt.figure(figsize=(10, 8))
        plt.imshow(erf_matrix, cmap='hot')
        plt.colorbar(label='Gradient Contribution')
        plt.title('Effective Receptive Field')
        
        # 保存可视化结果
        heatmap_path = os.path.join(model_dir, "heatmap.png")
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 读取结果图像
        with open(heatmap_path, 'rb') as f:
            result_image = base64.b64encode(f.read()).decode('utf-8')
        
        # 清理临时文件
        import shutil
        shutil.rmtree(model_dir)
        
        # 返回结果
        result = {
            "task_id": task_id,
            "status": "success",
            "result_image": result_image
        }
        
    except Exception as e:
        result = {
            "task_id": task_id,
            "status": "error",
            "message": str(e)
        }
    
    # 将结果放入队列
    result_queue.put(result)
    # 更新处理状态
    processing_status[task_id] = "completed"

# 路由：初始化模型
@app.route('/api/init_model', methods=['GET'])
def init_model():
    try:
        init_detection_model()
        return jsonify({"status": "success", "message": "模型初始化成功"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

# 路由：开始图像检测
@app.route('/api/detect', methods=['POST'])
def detect_image():
    try:
        data = request.json
        image_data = data.get('image_data')
        task_id = data.get('task_id')
        
        if not image_data or not task_id:
            return jsonify({"status": "error", "message": "缺少必要参数"})
        
        # 检查模型是否已初始化
        if detection_model is None:
            init_detection_model()
        
        # 更新处理状态
        processing_status[task_id] = "processing"
        
        # 直接同步执行检测，而不是启动线程
        # 解码图像数据
        img = Image.open(io.BytesIO(base64.b64decode(image_data)))
        img = img.convert('RGB')
        
        # 图像预处理
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((256, 256)),
            torchvision.transforms.ToTensor()
        ])
        
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        # 模型推理
        with torch.no_grad():
            output = detection_model(img_tensor)
            fake_conf = output[0][0].item()
            real_conf = output[0][1].item()
        
        # 直接返回结果，不使用队列
        result = {
            "task_id": task_id,
            "status": "success",
            "is_fake": fake_conf > real_conf,
            "fake_confidence": fake_conf,
            "real_confidence": real_conf
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

# 路由：开始换脸
@app.route('/api/face_swap', methods=['POST'])
def face_swap():
    try:
        data = request.json
        source_image = data.get('source_image')
        target_image = data.get('target_image')
        
        if not source_image or not target_image:
            return jsonify({"status": "error", "message": "缺少必要参数"})
        
        # 直接同步执行换脸，而不是启动线程
        # 创建临时文件
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as source_file, \
             tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as target_file, \
             tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as output_file:
            
            # 保存图像数据到临时文件
            source_file.write(base64.b64decode(source_image))
            target_file.write(base64.b64decode(target_image))
            
            # 执行换脸
            run_face_swapping(
                source_file.name, 
                target_file.name, 
                output_file.name
            )
            
            # 读取结果图像
            with open(output_file.name, 'rb') as f:
                result_image = base64.b64encode(f.read()).decode('utf-8')
        
        # 清理临时文件
        os.unlink(source_file.name)
        os.unlink(target_file.name)
        os.unlink(output_file.name)
        
        # 直接返回结果
        result = {
            "status": "success",
            "result_image": result_image
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})
    

# 路由：上传模型
@app.route('/api/upload_model', methods=['POST'])
def upload_model():
    try:
        data = request.json
        model_file = data.get('model_file')
        task_id = data.get('task_id')
        
        if not model_file or not task_id:
            return jsonify({"status": "error", "message": "缺少必要参数"})
        
        # 保存模型文件到临时目录
        model_dir = os.path.join(tempfile.gettempdir(), f"model_{task_id}")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "model.pth")
        
        with open(model_path, 'wb') as f:
            f.write(base64.b64decode(model_file))
        
        # 更新处理状态
        processing_status[task_id] = "model_uploaded"
        
        return jsonify({"status": "success", "task_id": task_id, "message": "模型上传成功"})
    
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})



# 路由：生成热力图
@app.route('/api/generate_heatmap', methods=['POST'])
def generate_heatmap():
    try:
        data = request.json
        task_id = data.get('task_id')
        if not task_id:
            return jsonify({"status": "error", "message": "缺少必要参数"})

        # 获取模型文件
        model_dir = os.path.join(tempfile.gettempdir(), f"model_{task_id}")
        model_path = os.path.join(model_dir, "model.pth")
        if not os.path.exists(model_path):
            return jsonify({"status": "error", "message": "模型文件不存在"})

        # —— 从固定文件夹读取测试图像 —— #
        # 项目根目录下 heatmap_images 中放 50 张图片
        src_folder = r"F:\Software_Engineering\Deepfake-Detection-new\data\archive\test\real"
        image_paths = sorted(
            glob.glob(os.path.join(src_folder, "*.jpg")) +
            glob.glob(os.path.join(src_folder, "*.png"))
        )[:10]  # 只取前 10 张

        if len(image_paths) == 0:
            return jsonify({"status": "error", "message": "heatmap_images 目录下没有找到图片"})
        
        # 清空旧目录（如果有），然后按照 visualize_erf.py 的要求组织目录
        valid_dir = os.path.join(model_dir, "valid")
        class_dir = os.path.join(valid_dir, "test_class")
        if os.path.exists(valid_dir):
            shutil.rmtree(valid_dir)
        os.makedirs(class_dir, exist_ok=True)

        # 依次复制 50 张图片到 class_dir
        for idx, img_path in enumerate(image_paths, start=1):
            ext = os.path.splitext(img_path)[1]
            dst_name = f"img_{idx:03d}{ext}"
            shutil.copy(img_path, os.path.join(class_dir, dst_name))

        # —— 调用 visualize_erf —— #
        args = parse_args()
        args.model_path = model_path
        args.data_path = model_dir        # visualize_erf.py 会在这里寻找 valid/
        args.save_path = os.path.join(model_dir, "erf.npy")
        args.num_images = len(image_paths)  # 通常是 50

        print(f"[Heatmap] 从 {src_folder} 复制了 {len(image_paths)} 张图到 {class_dir}")
        print(f"[Heatmap] 调用参数: data_path={args.data_path}, num_images={args.num_images}")

        visualize_erf_main(args)

        # 读取并可视化
        erf_matrix = np.load(args.save_path)
        plt.figure(figsize=(10, 8))
        plt.imshow(erf_matrix, cmap='hot')
        plt.colorbar(label='Gradient Contribution')
        plt.title('Effective Receptive Field')

        heatmap_path = os.path.join(model_dir, "heatmap.png")
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()

        with open(heatmap_path, 'rb') as f:
            result_image = base64.b64encode(f.read()).decode('utf-8')

        return jsonify({
            "task_id": task_id,
            "status": "success",
            "result_image": result_image
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

# 路由：获取处理结果
@app.route('/api/get_result', methods=['GET'])
def get_result():
    task_id = request.args.get('task_id')
    
    if not task_id:
        return jsonify({"status": "error", "message": "缺少任务ID"})
    
    # 直接检查处理状态
    status = processing_status.get(task_id, "unknown")
    
    if status == "completed":
        # 从结果队列中查找特定任务的结果
        with result_queue.mutex:  # 加锁确保线程安全
            for result in list(result_queue.queue):
                if result.get("task_id") == task_id:
                    return jsonify(result)
        
        return jsonify({
            "status": "error",
            "message": "结果已过期"
        })
    
    elif status == "processing":
        return jsonify({
            "status": "processing",
            "message": "结果尚未准备好"
        })
    
    else:
        return jsonify({
            "status": "error",
            "message": "无效的任务ID"
        })

# 启动服务器
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AI Backend Server')
    parser.add_argument('--host', default='0.0.0.0', help='服务器主机地址')
    parser.add_argument('--port', default=5000, type=int, help='服务器端口')
    parser.add_argument('--debug', action='store_true', help='是否开启调试模式')
    
    args = parser.parse_args()
    
    print(f"服务器启动中... 主机: {args.host}, 端口: {args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)
    
