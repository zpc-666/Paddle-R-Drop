# coding=utf-8
from __future__ import division, print_function
from utils import logger, AverageMeter, simple_accuracy, setup, get_loader
from tqdm import tqdm
import paddle
import numpy as np
import os

def valid(args, model, test_loader, global_step):
    # Validation!
    eval_losses = AverageMeter()

    logger.info("***** Running Validation *****")
    logger.info("  Num steps = %d", len(test_loader))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    all_preds, all_label = [], []

    loss_fct = paddle.nn.CrossEntropyLoss()
    for step, batch in enumerate(test_loader):
        x, y = batch
        with paddle.no_grad():
            logits = model(x)[0]

            eval_loss = loss_fct(logits, y)
            eval_losses.update(eval_loss.item())

            preds = paddle.argmax(logits, axis=-1)

        if len(all_preds) == 0:
            all_preds.append(preds.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())
        else:
            all_preds[0] = np.append(
                    all_preds[0], preds.detach().cpu().numpy(), axis=0
            )
            all_label[0] = np.append(
                    all_label[0], y.detach().cpu().numpy(), axis=0
            )
        if step%30==0:
            print("Validating... (loss=%2.5f(%2.5f))" % (eval_losses.val, eval_losses.avg))

    all_preds, all_label = all_preds[0], all_label[0]
    accuracy = simple_accuracy(all_preds, all_label)

    logger.info("\n")
    logger.info("Validation Results")
    logger.info("Global Steps: %d" % global_step)
    logger.info("Valid Loss: %2.5f" % eval_losses.avg)
    logger.info("Valid Accuracy: %2.5f" % accuracy)

    return accuracy, eval_losses.avg


if __name__ == '__main__':

    # Model & Tokenizer Setup
    args, model = setup()

    if args.checkpoint_dir is not None and os.path.exists(args.checkpoint_dir):
        model.set_state_dict(paddle.load(args.checkpoint_dir))
        # Prepare dataset
        train_loader, test_loader = get_loader(args)
        accuracy = valid(args, model, test_loader, global_step=0)
        print("Accuracy:", accuracy)
    else:
        raise ValueError("checkpoint_dir must exist.")
