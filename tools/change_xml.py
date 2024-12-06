'''
Author: aliyun9951438140 huangquanjin24@gmail.com
Date: 2024-12-06 14:31:56
LastEditors: aliyun9951438140 huangquanjin24@gmail.com
LastEditTime: 2024-12-06 14:34:43
FilePath: /2-D-Kalman-Filter/change_xml.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import os
import xml.etree.ElementTree as ET

def change_xml_filename(xml_path):
    """
    修改XML文件中的filename标签内容，将.xml后缀改为.jpg
    :param xml_path: XML文件路径
    """
    try:
        # 解析XML文件
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # 查找filename标签
        filename_elem = root.find('filename')
        if filename_elem is not None:
            # 获取当前文件名
            current_filename = filename_elem.text
            # 如果文件名以.xml结尾，替换为.jpg
            if current_filename.endswith('.xml'):
                new_filename = current_filename[:-4] + '.jpg'
                filename_elem.text = new_filename
                # 保存修改后的XML文件
                tree.write(xml_path, encoding='utf-8', xml_declaration=True)
                print(f"已修改 {xml_path} 中的文件名从 {current_filename} 到 {new_filename}")
            else:
                print(f"文件 {xml_path} 中的文件名不以.xml结尾")
        else:
            print(f"在 {xml_path} 中未找到filename标签")
            
    except ET.ParseError as e:
        print(f"解析XML文件 {xml_path} 时出错: {e}")
    except Exception as e:
        print(f"处理文件 {xml_path} 时发生错误: {e}")

def process_directory(directory):
    """
    处理指定目录下的所有XML文件
    :param directory: 要处理的目录路径
    """
    for filename in os.listdir(directory):
        if filename.endswith('.xml'):
            xml_path = os.path.join(directory, filename)
            change_xml_filename(xml_path)

if __name__ == '__main__':
    # 获取当前目录
    voc_dir = '/data1/DATA_126/hqj/MAR20/train/voc'
    process_directory(voc_dir)
