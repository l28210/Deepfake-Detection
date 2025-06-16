#!/usr/bin/env python3

import sys
from roop import core
import roop.globals
import argparse
from PIL import Image, ImageDraw, ImageFont  # 添加PIL库的导入
import os

def add_watermark(image_path):
    """在图片右下角添加文字水印"""
    try:
        # 打开图片
        with Image.open(image_path).convert("RGBA") as img:
            width, height = img.size
            
            # 创建一个透明的画布用于绘制文字
            text_layer = Image.new('RGBA', img.size, (255, 255, 255, 0))
            draw = ImageDraw.Draw(text_layer)
            
            # 设置文字内容和字体
            text = "图片由AI生成"
            # 尝试加载系统字体，或者使用默认字体
            try:
                # Windows系统字体路径
                font_path = "C:/Windows/Fonts/simhei.ttf"
                if not os.path.exists(font_path):
                    # Linux系统字体路径
                    font_path = "/usr/share/fonts/truetype/wqy/wqy-microhei.ttf"
                font = ImageFont.truetype(font_path, 28)  # 增大字体大小至28
            except (IOError, OSError):
                # 如果找不到中文字体，使用默认字体并尝试增大尺寸
                font = ImageFont.load_default()
                # 某些默认字体可能不支持指定大小，尝试使用font.font_variant
                try:
                    font = font.font_variant(size=28)
                except AttributeError:
                    pass  # 使用默认大小

            # 计算文字尺寸（使用textbbox代替textsize）
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            # 设置文字位置（右下角，留出15px边距，增加间距）
            position = (width - text_width - 15, height - text_height - 15)

            # 绘制文字（深白色，使用接近白色但仍有辨识度的颜色）
            draw.text(position, text, font=font, fill=(240, 240, 240, 255))  # 深白色，不透明度100%
            
            # 合并原图和文字层
            watermarked_img = Image.alpha_composite(img.convert('RGBA'), text_layer)
            
            # 保存图片（如果是JPG格式，需要转换回RGB）
            if image_path.lower().endswith(('.jpg', '.jpeg')):
                watermarked_img = watermarked_img.convert('RGB')
            watermarked_img.save(image_path)
            
            print(f"已在图片 {image_path} 右下角添加水印")
            
    except Exception as e:
        print(f"添加水印时出错: {e}")

def run_face_swapping(source_image_path, target_image_path, output_image_path):
    # 设置全局变量
    roop.globals.source_path = source_image_path
    roop.globals.target_path = target_image_path
    roop.globals.output_path = output_image_path
    roop.globals.headless = True  # 以无头模式运行，不显示 UI

    # 手动创建命令行参数对象
    args = argparse.Namespace(
        source_path=source_image_path,
        target_path=target_image_path,
        output_path=output_image_path,
        frame_processor=['face_swapper'],
        keep_fps=False,
        keep_frames=False,
        skip_audio=False,
        many_faces=False,
        reference_face_position=0,
        reference_frame_number=0,
        similar_face_distance=0.85,
        temp_frame_format='png',
        temp_frame_quality=0,
        output_video_encoder='libx264',
        output_video_quality=35,
        max_memory=None,
        execution_provider=['cpu'],
        execution_threads=1
    )
    roop.globals.source_path = args.source_path
    roop.globals.target_path = args.target_path
    roop.globals.output_path = args.output_path
    roop.globals.frame_processors = args.frame_processor
    roop.globals.keep_fps = args.keep_fps
    roop.globals.keep_frames = args.keep_frames
    roop.globals.skip_audio = args.skip_audio
    roop.globals.many_faces = args.many_faces
    roop.globals.reference_face_position = args.reference_face_position
    roop.globals.reference_frame_number = args.reference_frame_number
    roop.globals.similar_face_distance = args.similar_face_distance
    roop.globals.temp_frame_format = args.temp_frame_format
    roop.globals.temp_frame_quality = args.temp_frame_quality
    roop.globals.output_video_encoder = args.output_video_encoder
    roop.globals.output_video_quality = args.output_video_quality
    roop.globals.max_memory = args.max_memory
    roop.globals.execution_providers = core.decode_execution_providers(args.execution_provider)
    roop.globals.execution_threads = args.execution_threads

    # 进行前置检查
    if not core.pre_check():
        print("Pre-check failed. Exiting...")
        return

    # 限制资源使用
    core.limit_resources()

    # 开始处理
    core.start()
    
    # 处理完成后添加水印
    if os.path.exists(output_image_path):
        add_watermark(output_image_path)
    else:
        print(f"警告: 输出图片 {output_image_path} 不存在，无法添加水印")

if __name__ == '__main__':
    source_image_path = r"F:\Software_Engineering\Deepfake-Detection-new\lib\Roop\soure\003.jpg"
    target_image_path = r"F:\Software_Engineering\Deepfake-Detection-new\lib\Roop\soure\002.jpg"
    output_image_path = r"F:\Software_Engineering\Deepfake-Detection-new\lib\Roop\result\result001.jpg"

    run_face_swapping(source_image_path, target_image_path, output_image_path)