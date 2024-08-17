import os
import xml.etree.ElementTree as ET
import argparse

def convert_voc_to_yolo(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    #Load class names from a file called
    classes_file = os.path.join(input_dir, 'classes.txt')
    if not os.path.exists(classes_file):
        raise FileNotFoundError(f"classes.txt not found in {input_dir}")
    
    with open(classes_file, 'r') as f:
        classes = f.read().strip().split()

    for xml_file in os.listdir(os.path.join(input_dir, 'pascalVOC_labels')):
        if xml_file.endswith('.xml'):
            tree = ET.parse(os.path.join(input_dir, 'pascalVOC_labels', xml_file))
            root = tree.getroot()
            image_id = os.path.splitext(xml_file)[0]

            with open(os.path.join(output_dir, f'{image_id}.txt'), 'w') as yolo_file:
                for obj in root.iter('object'):
                    class_name = obj.find('name').text
                    if class_name in classes:
                        class_id = classes.index(class_name)
                        xmlbox = obj.find('bndbox')
                        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text),
                             float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
                        b = convert_bbox((int(root.find('size/width').text), int(root.find('size/height').text)), b)
                        yolo_file.write(f"{class_id} " + " ".join([str(a) for a in b]) + '\n')

def convert_bbox(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert PascalVOC annotations to YOLOv8 format")
    parser.add_argument('input_dir', type=str, help="Base input directory path containing images and labels")
    parser.add_argument('output_dir', type=str, help="Output directory path where YOLOv8 annotations will be saved")
    
    args = parser.parse_args()

    convert_voc_to_yolo(args.input_dir, args.output_dir)
