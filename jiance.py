import xml.etree.ElementTree as ET
import os

# 1. 定义类别映射（根据实际类别修改）
class_mapping = {
    "holothurian": 0,
    "echinus": 1,
    "scallop":2,
}

def xml_to_yolo(xml_path, output_dir):
    """将单个XML文件转换为YOLO格式的TXT文件"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # 2. 获取图像尺寸
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    
    # 3. 生成输出文件名（与XML同名但后缀为.txt）
    xml_name = os.path.basename(xml_path)
    txt_name = os.path.splitext(xml_name)[0] + ".txt"
    txt_path = os.path.join(output_dir, txt_name)
    
    # 4. 解析每个目标对象
    with open(txt_path, 'w') as f:
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            # 跳过未定义类别
            if class_name not in class_mapping:
                continue
            
            # 获取边界框坐标
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            
            # 5. 计算归一化坐标
            x_center = (xmin + xmax) / (2 * width)
            y_center = (ymin + ymax) / (2 * height)
            bbox_width = (xmax - xmin) / width
            bbox_height = (ymax - ymin) / height
            
            # 6. 写入TXT文件
            f.write(f"{class_mapping[class_name]} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")

# 7. 批量转换（示例路径）
xml_dir = "D:/testprogram/yolov5-7.0/datasets/coco128/images/mmm"     # XML文件夹路径
output_dir = "D:/testprogram/yolov5-7.0/datasets/coco128/labels/myf"       # 输出TXT文件夹
os.makedirs(output_dir, exist_ok=True)

for xml_file in os.listdir(xml_dir):
    if xml_file.endswith(".xml"):
        xml_path = os.path.join(xml_dir, xml_file)
        xml_to_yolo(xml_path, output_dir)
