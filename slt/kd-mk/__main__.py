import argparse
import os

import sys
from signjoey.training import train
from signjoey.prediction import test
from signjoey.kd import convertModel
from signjoey.helpers import (
    log_data_info,
    load_config,
    log_cfg,
    load_checkpoint,
    make_model_dir,
    make_logger,
    set_seed,
    symlink_update,
)
from signjoey.data import load_data, make_data_iter
from signjoey.model import load_teacher_model

sys.path.append("/vol/research/extol/personal/cihan/code/SignJoey")


def main():
    ap = argparse.ArgumentParser("Joey NMT")

    ap.add_argument("mode", choices=["train", "test", "convert"], help="train a model or test")

    ap.add_argument("config_path", type=str, help="path to YAML config file")

    ap.add_argument("--ckpt", type=str, help="checkpoint for prediction")

    ap.add_argument(
        "--output_path", type=str, help="path for saving translation output"
    )
    ap.add_argument("--gpu_id", type=str, default="0", help="gpu to run your job on")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    if args.mode == "train" or args.mode == "train":
        # CONFIG FOR TEACHER MODEL
        cfg_file = "configs/fsl-tfl/combined.yaml"
    
        cfg = load_config(cfg_file)
        train_data, dev_data, test_data, gls_vocab, txt_vocab = load_data(data_cfg=cfg["data"]) 
        teacher_model = load_teacher_model(args.ckpt, cfg_file, gls_vocab, txt_vocab)

        if args.mode == "train":
            train(cfg_file=args.config_path, teacher=teacher_model)
        else: #test
            if args.output_path is not None:
                test(cfg_file=args.config_path, ckpt=args.ckpt, teacher=teacher_model, output_path=args.output_path)
            else:
                output_path=cfg["training"]["model_dir"]
                logger=make_logger(model_dir=output_path, log_file="test.log")
                ckpt = os.path.join(output_path, "best.ckpt")
                test(cfg_file=args.config_path, ckpt=ckpt, teacher=teacher_model, output_path=output_path, logger=logger)
    else:
        if args.mode == "convert":
            convertModel(cfg_file=args.config_path, ckpt=args.ckpt)
        else:
            raise ValueError("Unknown mode")


if __name__ == "__main__":
    main()
