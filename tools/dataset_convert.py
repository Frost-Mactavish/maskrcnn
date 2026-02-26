"""
directory structure of VOC-style dataset:
VOC2007/
├── Annotations/    # XML annotation files
│   ├── 000001.xml
│   ├── 000002.xml
│   └── ...
└── JPEGImages/     # images
│   ├── 000001.jpg
│   ├── 000002.jpg
│   └── ...
├── ImageSets/      # train/test split files
    ├── train.txt   # overall train/test split files
    ├── test.txt
    └── Main/
        ├── class1_train.txt    # per-class train/test split files
        ├── class1_test.txt
        ├── ...
        ├── classN_train.txt
        ├── classN_test.txt
        ├── train.txt   # overall train/test split files (copy of ../train.txt)
        └── test.txt

This script converts DOTA txt annotations to XML format,
and generates per-class & overall train/test split files for DIOR/DOTA
"""

import os
from glob import glob
from xml.etree.ElementTree import Element, SubElement, tostring
import xml.etree.ElementTree as ET
from xml.dom.minidom import parseString
from shutil import copy2


def get_data(xml_path):
    def parse_xml_to_dict(xml):
        if len(xml) == 0:
            return {xml.tag: xml.text}

        result = {}
        for child in xml:
            child_result = parse_xml_to_dict(child)
            if child.tag != "object":
                result[child.tag] = child_result[child.tag]
            else:
                if child.tag not in result:
                    result[child.tag] = []
                result[child.tag].append(child_result[child.tag])
        return {xml.tag: result}

    with open(xml_path, "r") as f:
        xml_str = f.read()
    xml = ET.fromstring(xml_str)

    return parse_xml_to_dict(xml)["annotation"]


if __name__ == "__main__":
    prefix_path = ""
    source_root = f"{prefix_path}/dataset/DOTA"
    target_root = f"{prefix_path}/dataset/DOTA_xml"

    target_anno_dir = f"{target_root}/Annotations"
    target_img_dir = f"{target_root}/JPEGImages"
    target_txt_dir = f"{target_root}/ImageSets/Main"

    os.makedirs(target_anno_dir, exist_ok=True)
    os.makedirs(target_img_dir, exist_ok=True)
    os.makedirs(target_txt_dir, exist_ok=True)

    # convert DOTA txt annotations to XML format
    source_label_dirs = (
        f"{source_root}/testsplit/labelTxt",
        f"{source_root}/trainsplit/labelTxt",
    )

    for label_dir in source_label_dirs:
        for txt in os.listdir(label_dir):
            if not txt.endswith(".txt"):
                continue

            img_name = txt.replace(".txt", ".png")
            width, height = 1024, 1024

            prefix = Element("annotation")
            SubElement(prefix, "filename").text = img_name
            size = SubElement(prefix, "size")
            SubElement(size, "width").text = str(width)
            SubElement(size, "height").text = str(height)
            SubElement(size, "depth").text = "3"

            with open(os.path.join(label_dir, txt)) as f:
                for line in f:
                    data = line.strip().split()
                    if len(data) < 10:
                        continue

                    x = list(map(float, data[:8]))
                    cls = data[8]
                    diff = data[9]

                    xs = x[0::2]
                    ys = x[1::2]
                    xmin, ymin = int(min(xs)), int(min(ys))
                    xmax, ymax = int(max(xs)), int(max(ys))

                    obj = SubElement(prefix, "object")
                    SubElement(obj, "name").text = cls
                    SubElement(obj, "difficult").text = diff

                    bnd = SubElement(obj, "bndbox")
                    SubElement(bnd, "xmin").text = str(xmin)
                    SubElement(bnd, "ymin").text = str(ymin)
                    SubElement(bnd, "xmax").text = str(xmax)
                    SubElement(bnd, "ymax").text = str(ymax)

            xml_str = parseString(tostring(prefix)).toprettyxml()
            xml_path = os.path.join(target_anno_dir, txt.replace(".txt", ".xml"))
            with open(xml_path, "w", encoding="utf-8") as f:
                f.write(xml_str)

    train_img_dir = os.path.join(source_root, "trainsplit/images")
    test_img_dir = os.path.join(source_root, "testsplit/images")

    # copy images to target directory
    for img_path in glob(os.path.join(train_img_dir, "*")):
        copy2(img_path, target_img_dir)
    for img_path in glob(os.path.join(test_img_dir, "*")):
        copy2(img_path, target_img_dir)

    # generate overall trainval.txt and test.txt for dataset
    train_img_list = [
        os.path.basename(f) for f in glob(os.path.join(train_img_dir, "*"))
    ]
    test_img_list = [os.path.basename(f) for f in glob(os.path.join(test_img_dir, "*"))]

    with open(os.path.join(target_txt_dir, "trainval.txt"), "w") as f:
        for img_name in train_img_list:
            f.write(f"{img_name[:-4]}\n")
    with open(os.path.join(target_txt_dir, "test.txt"), "w") as f:
        for img_name in test_img_list:
            f.write(f"{img_name[:-4]}\n")

    # generate per-class train/test split files for dataset
    # based on overall train/test split files and XML annotations
    # and save them in {dataset_root}/ImageSets/Main/ directory

    # class_name_list = [
    #     "airplane", "baseballfield", "bridge", "groundtrackfield", "vehicle",
    #     "ship", "tenniscourt", "airport", "chimney", "dam",
    #     "basketballcourt", "Expressway-Service-area", "Expressway-toll-station", "golffield", "harbor",
    #     "overpass", "stadium", "storagetank", "trainstation", "windmill",
    # ]

    class_name_list = [
        "airplane",
        "baseballfield",
        "bridge",
        "groundtrackfield",
        "vehicle",
        "ship",
        "tenniscourt",
        "airport",
        "chimney",
        "dam",
        "basketballcourt",
        "Expressway-Service-area",
        "Expressway-toll-station",
        "golffield",
        "harbor",
        "overpass",
        "stadium",
        "storagetank",
        "trainstation",
        "windmill",
    ]

    with open(os.path.join(target_txt_dir, "trainval.txt"), "r") as f:
        train_list = [line.strip() for line in f]
    with open(os.path.join(target_txt_dir, "test.txt"), "r") as f:
        test_list = [line.strip() for line in f]

    for category in class_name_list:
        for train_img in train_list:
            xml_path = os.path.join(target_root, "Annotations", train_img + ".xml")
            objects_in_img = get_data(xml_path)["object"]
            found = any(obj["name"] == category for obj in objects_in_img)
            out_path = os.path.join(target_txt_dir, category + "_train.txt")
            with open(out_path, "a") as f:
                f.write(f"{train_img} {(1 if found else -1):2d}\n")

        for test_img in test_list:
            xml_path = os.path.join(target_root, "Annotations", test_img + ".xml")
            objects_in_img = get_data(xml_path)["object"]
            found = any(obj["name"] == category for obj in objects_in_img)
            out_path = os.path.join(target_txt_dir, category + "_test.txt")
            with open(out_path, "a") as f:
                f.write(f"{test_img} {(1 if found else -1):2d}\n")
