# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

from .wfm import wfm_base, wfm_small, wfm_tiny


def build_model(config):
    model_type = config.MODEL.TYPE
    model_name = config.MODEL.NAME
    if model_type == "wfm":
        model = eval(
            model_name + "(img_size=config.DATA.IMG_SIZE,"
            "drop_path_rate=config.MODEL.DROP_PATH_RATE)"
        )
    else:
        raise NotImplementedError(f"Unknown model: {model_type}")
    return model
