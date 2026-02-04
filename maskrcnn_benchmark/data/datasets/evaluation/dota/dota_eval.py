import os

import torch

from .dota_utils import merge_and_eval


def do_dota_evaluation(dataset, predictions, file_list, output_folder, logger, cfg):
    class_list = cfg.MODEL.ROI_BOX_HEAD.NAME_OLD_CLASSES + cfg.MODEL.ROI_BOX_HEAD.NAME_NEW_CLASSES
    class_txt = [[] for _ in class_list]

    for file, prediction in zip(file_list, predictions):
        prediction = prediction.resize((1024, 1024))
        result = {'boxes': prediction.bbox, 'labels': prediction.get_field("labels"),
                  'scores': prediction.get_field("scores")}
        for b, l, s in zip(*result.values()):
            b = torch.tensor([b[0], b[1], b[2], b[1], b[2], b[3], b[0], b[3]])
            bbox_str = ' '.join([str(x) for x in b.tolist()])
            class_txt[l.item() - 1].append(f"{file} {s.item()} {bbox_str}\n")

    path = os.path.join(output_folder, "DOTA_result")
    os.makedirs(path, exist_ok=True)
    for class_name, txt in zip(class_list, class_txt):
        with open(f"{path}/{class_name}.txt", "w") as f:
            f.writelines(txt)
    print_msg, map = merge_and_eval(path, path, class_list)
    print("\n".join(print_msg))

    if output_folder:
        with open(os.path.join(output_folder, "result.txt"), "w") as f:
            f.write("\n".join(print_msg))

    return map