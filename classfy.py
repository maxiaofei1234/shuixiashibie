import os
import shutil
from pathlib import Path

# ====== 配置路径 ======
label_dir = Path("D:/testprogram/yolov5-7.0/datasets/coco128/labels/train")  # 标签文件夹
image_dir = Path("D:/testprogram/yolov5-7.0/datasets/coco128/images/train")  # 图片文件夹
output_image_dir = Path("D:/testprogram/yolov5-7.0/datasets/coco128/images/classified_images")  # 图片输出目录
output_label_dir = Path("D:/testprogram/yolov5-7.0/datasets/coco128/labels/新建文件夹")  # 标签输出目录

# ====== 创建输出目录 ======
output_image_dir.mkdir(parents=True, exist_ok=True)
output_label_dir.mkdir(parents=True, exist_ok=True)

# ====== 核心分类逻辑 ======
for label_file in label_dir.glob("*.txt"):
    # 提取类别ID（标签文件首行首个字段）
    with open(label_file, 'r') as f:
        first_line = f.readline().strip()
        if not first_line:  # 跳过空标签文件
            continue
        class_id = first_line.split()[0]  # 示例: "0" -> class_id="0"
    
    # 创建类别专属的图片/标签目录
    class_image_dir = output_image_dir / class_id
    class_label_dir = output_label_dir / class_id
    class_image_dir.mkdir(exist_ok=True)
    class_label_dir.mkdir(exist_ok=True)

    # ====== 处理图片文件 ======
    img_name = label_file.stem + ".jpg"  # 假设图片为jpg格式
    src_img_path = image_dir / img_name
    if src_img_path.exists():
        shutil.copy(src_img_path, class_image_dir / img_name)
    else:
        print(f"⚠️ 图片缺失: {img_name}")

    # ====== 处理标签文件 ======
    target_label_path = class_label_dir / label_file.name
    shutil.copy(label_file, target_label_path)  # 复制标签到对应类别目录

print("✅ 分类完成！")
print(f"图片输出至: {output_image_dir}")
print(f"标签输出至: {output_label_dir}")
