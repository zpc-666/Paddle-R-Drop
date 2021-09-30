# coding=utf-8
from __future__ import absolute_import, division, print_function

import paddle
from utils import setup, get_loader
from train import train_model
from evaluate import valid
import os

def main():
    
    # Model & Tokenizer Setup
    args, model = setup()

    if args.mode=='train':
        # Training
        train_model(args, model)
    elif args.mode=='eval':
        if args.checkpoint_dir is not None and os.path.exists(args.checkpoint_dir):
            model.set_state_dict(paddle.load(args.checkpoint_dir))
            # Prepare dataset
            train_loader, test_loader = get_loader(args)
            accuracy = valid(args, model, test_loader, global_step=0)
            print("Accuracy:",accuracy)
        else:
            raise ValueError("checkpoint_dir must exist.")
    else:
        raise ValueError("mode must be in ['train', 'eval'].")

if __name__ == "__main__":
    main()
