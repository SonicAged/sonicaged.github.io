import os
import fitz  # PyMuPDF
import argparse
from pathlib import Path
import hashlib
import numpy as np
from PIL import Image
import io

def is_blank_image(image_bytes, threshold=0.95):
    """检查图像是否全黑或全白"""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img_array = np.array(img.convert("L"))
        dark_ratio = np.sum(img_array < 10) / img_array.size
        light_ratio = np.sum(img_array > 245) / img_array.size
        return dark_ratio > threshold or light_ratio > threshold
    except:
        return False

def extract_images_from_pdf(pdf_path, output_folder, prefix="", min_width=100, min_height=100):
    """从PDF中提取图片并保存到输出文件夹"""
    pdf_document = fitz.open(pdf_path)
    image_count = 0
    
    # 方法1: 提取嵌入图像
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        images = page.get_images(full=True)
        
        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]
            
            # 检查图像尺寸
            try:
                img = Image.open(io.BytesIO(image_bytes))
                if img.width < min_width or img.height < min_height:
                    continue
            except:
                continue
                
            # 检查是否是空白图像
            if is_blank_image(image_bytes):
                continue
            
            # 使用哈希值作为文件名的一部分，避免重名
            image_hash = hashlib.md5(image_bytes).hexdigest()[:10]
            image_ext = base_image["ext"]
            image_filename = f"{prefix}_embedded_p{page_num+1}_i{img_index+1}_{image_hash}.{image_ext}"
            
            with open(os.path.join(output_folder, image_filename), "wb") as image_file:
                image_file.write(image_bytes)
            
            image_count += 1
    
    # 方法2: 渲染整页或识别图形区域
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        
        # 渲染整页为高清图片
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        image_bytes = pix.tobytes("png")
        
        # 跳过空白页面
        if is_blank_image(image_bytes, threshold=0.98):
            continue
            
        image_filename = f"{prefix}_rendered_p{page_num+1}.png"
        with open(os.path.join(output_folder, image_filename), "wb") as image_file:
            image_file.write(image_bytes)
        
        image_count += 1
    
    pdf_document.close()
    return image_count

def main():
    parser = argparse.ArgumentParser(description="从一个文件夹中的所有PDF提取图片")
    parser.add_argument("input_dir", help="包含PDF文件的输入目录")
    parser.add_argument("output_dir", help="保存图片的输出目录")
    parser.add_argument("--min-width", type=int, default=100, help="最小图片宽度")
    parser.add_argument("--min-height", type=int, default=100, help="最小图片高度")
    parser.add_argument("--extract-mode", choices=["embedded", "rendered", "both"], default="embedded",
                       help="提取模式: embedded(嵌入图像), rendered(渲染页面), both(两者都要)")
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    total_pdfs = 0
    total_images = 0
    
    # 递归遍历输入目录
    for root, _, files in os.walk(args.input_dir):
        for file in files:
            if file.lower().endswith(".pdf"):
                pdf_path = os.path.join(root, file)
                pdf_name = os.path.splitext(os.path.basename(file))[0]
                
                print(f"正在处理 PDF: {pdf_path}")
                
                try:
                    # 提取图片并使用PDF文件名作为前缀
                    image_count = extract_images_from_pdf(pdf_path, args.output_dir, 
                                                         prefix=pdf_name,
                                                         min_width=args.min_width, 
                                                         min_height=args.min_height)
                    print(f"已从 {pdf_path} 中提取 {image_count} 张图片")
                    
                    total_pdfs += 1
                    total_images += image_count
                except Exception as e:
                    print(f"处理 {pdf_path} 时出错: {str(e)}")
    
    print(f"\n处理完成! 总共处理了 {total_pdfs} 个PDF文件，提取了 {total_images} 张图片。")
    print(f"所有图片已保存到: {os.path.abspath(args.output_dir)}")

if __name__ == "__main__":
    main()