import os
import fitz
from pathlib import Path


def pdf2img(pdf_path, output_dir=None, zoom_x=3, zoom_y=3):
    """
    将单个PDF文件转换为PNG图片
    
    Args:
        pdf_path (str): PDF文件的路径
        output_dir (str, optional): 输出目录，默认与PDF在同一目录
        zoom_x (int): X轴缩放比例
        zoom_y (int): Y轴缩放比例
    """
    # 获取PDF文件名（不含扩展名）
    pdf_name = Path(pdf_path).stem
    
    # 如果没有指定输出目录，使用PDF所在目录
    if output_dir is None:
        output_dir = str(Path(pdf_path).parent)
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        doc = fitz.open(pdf_path)  # 打开文档
        # 如果PDF只有一页，直接使用PDF文件名
        if doc.page_count == 1:
            page = doc[0]
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom_x, zoom_y))
            output_path = os.path.join(output_dir, f"{pdf_name}.png")
            pix.save(output_path)
        else:
            # 如果有多页，为每页添加页码
            for page in doc:
                pix = page.get_pixmap(matrix=fitz.Matrix(zoom_x, zoom_y))
                output_path = os.path.join(output_dir, f"{pdf_name}_page{page.number+1}.png")
                pix.save(output_path)
        doc.close()
        print(f"成功转换: {pdf_path}")
    except Exception as e:
        print(f"转换失败 {pdf_path}: {str(e)}")


def convert_folder(folder_path, output_dir=None, zoom_x=10, zoom_y=10):
    """
    转换指定文件夹中的所有PDF文件为PNG
    
    Args:
        folder_path (str): 包含PDF文件的文件夹路径
        output_dir (str, optional): 输出目录，默认与PDF在同一目录
        zoom_x (int): X轴缩放比例
        zoom_y (int): Y轴缩放比例
    """
    folder_path = Path(folder_path)
    
    # 如果指定了输出目录，确保它存在
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 遍历文件夹中的所有PDF文件
    for pdf_file in folder_path.glob("*.pdf"):
        pdf2img(str(pdf_file), output_dir, zoom_x, zoom_y)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="将PDF文件转换为PNG图片")
    parser.add_argument("path", help="PDF文件或包含PDF文件的文件夹的路径")
    parser.add_argument("-o", "--output", help="输出目录（可选）", default=None)
    parser.add_argument("-x", "--zoom-x", type=float, default=10, help="X轴缩放比例（默认：10）")
    parser.add_argument("-y", "--zoom-y", type=float, default=10, help="Y轴缩放比例（默认：10）")
    
    args = parser.parse_args()
    path = Path(args.path)
    
    if path.is_file() and path.suffix.lower() == '.pdf':
        # 如果输入是单个PDF文件
        pdf2img(str(path), args.output, args.zoom_x, args.zoom_y)
    elif path.is_dir():
        # 如果输入是文件夹
        convert_folder(str(path), args.output, args.zoom_x, args.zoom_y)
    else:
        print("错误：请提供有效的PDF文件或文件夹路径") 