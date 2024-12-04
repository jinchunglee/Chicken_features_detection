import os
import xml.etree.ElementTree as ET
import pandas as pd

# Function to convert XML files to a structured CSV
def convert_xml_to_csv(xml_folder, output_csv):
    """
    Convert XML annotation files to a CSV file.

    Args:
        xml_folder (str): The path to the folder containing XML files.
        output_csv (str): The path to save the resulting CSV file.
    """
    rows = []
    # 類別名稱映射字典（可自行修改）
    label_mapping = {
        "ChickenCrown": "crown",
        "ChickenFeet": "feet",
        "ChickenTail": "tail",
        "ChickenEye": "eyes"
    }

    for xml_file in os.listdir(xml_folder):
        if not xml_file.endswith('.xml'):
            continue
        xml_path = os.path.join(xml_folder, xml_file)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        img_filename = root.find('filename').text
        for obj in root.findall('object'):
            label = obj.find('name').text
            # 使用映射字典進行替換（若名稱不在字典中，保持原樣）
            label = label_mapping.get(label, label)
            bndbox = obj.find('bndbox')
            xmin = bndbox.find('xmin').text
            ymin = bndbox.find('ymin').text
            xmax = bndbox.find('xmax').text
            ymax = bndbox.find('ymax').text
            rows.append([img_filename, label, xmin, ymin, xmax, ymax])

    # Save rows to a CSV file
    df = pd.DataFrame(rows, columns=['filename', 'type', 'xmin', 'ymin', 'xmax', 'ymax'])
    df.to_csv(output_csv, index=False)
    print(f"CSV file saved to: {output_csv}")

# Function to convert CSV to TXT format
def convert_csv_to_txt(input_csv, output_txt, image_folder):
    """
    Convert a CSV file to a TXT file in the required annotation format.

    Args:
        input_csv (str): The path to the input CSV file.
        output_txt (str): The path to save the resulting TXT file.
        image_folder (str): The folder containing the image files.
    """
    df = pd.read_csv(input_csv)
    with open(output_txt, 'w') as txt_file:
        for _, row in df.iterrows():
            txt_line = f"{os.path.join(image_folder, row['filename'])},{row['xmin']},{row['ymin']},{row['xmax']},{row['ymax']},{row['type']}\n"
            txt_file.write(txt_line)
    print(f"TXT file saved to: {output_txt}")

# Main function
if __name__ == "__main__":
    # *** 修改這些路徑 ***
    xml_folder = "xml"  # XML 資料夾路徑
    output_csv = "./annotations.csv"  # 中間的 CSV 檔案存放路徑
    output_txt = "./annotate.txt"  # 最終的 TXT 檔案存放路徑
    image_folder = "images"  # 圖片所在的資料夾路徑

    # 確保輸出目錄存在
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    os.makedirs(os.path.dirname(output_txt), exist_ok=True)

    # 執行轉換
    convert_xml_to_csv(xml_folder, output_csv)
    convert_csv_to_txt(output_csv, output_txt, image_folder)
