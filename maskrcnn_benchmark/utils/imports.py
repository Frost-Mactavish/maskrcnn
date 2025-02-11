# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import imp

import torch


def import_file(module_name, file_path, make_importable=None):
    module = imp.load_source(module_name, file_path)
    return module
