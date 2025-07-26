import os
from pathlib import Path

def export_filenames_to_href(folder_path, output_file="hrefs.txt"):
    """
    将文件夹中所有论文写入TXT文件，并生成href链接
    :param folder_path: 目标文件夹路径
    :param output_file: 输出文件名（默认为hrefs.txt）
    """
    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        print(f"错误：文件夹 '{folder_path}' 不存在！")
        return

    # 获取所有文件名（无扩展名）
    file_names = []
    for entry in os.listdir(folder_path):
        full_path = os.path.join(folder_path, entry)
        if os.path.isfile(full_path):
            name_without_ext = Path(entry).stem  # 直接去除扩展名[9,10](@ref)
            href = f"<a href=\"/paper/CDR/2024/{entry}\" target=\"_blank\">📄 {name_without_ext}</a>\n"
            file_names.append(href)

    # 写入TXT文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for name in file_names:
            f.write(name + '\n')

    print(f"成功将 {len(file_names)} 个文件名写入 {output_file}")

# 使用示例
if __name__ == "__main__":
    target_folder = input("请输入文件夹路径：")
    export_filenames_to_href(target_folder)