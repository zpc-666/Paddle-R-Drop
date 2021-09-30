# coding=utf-8
from __future__ import absolute_import, division, print_function

import os
import paddle

from tqdm import tqdm
from visualdl import LogWriter

from utils import WarmupLinearSchedule, WarmupCosineSchedule, get_loader, logger, AverageMeter, setup, set_seed, save_model, resume
from evaluate import valid

def train_model(args, model):
    """ Train the model """
    os.makedirs(args.output_dir, exist_ok=True)
    writer = LogWriter(logdir=os.path.join("logs", args.name))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    # Prepare dataset
    train_loader, test_loader = get_loader(args)

    # Prepare optimizer and scheduler
    t_total = args.num_steps
    if args.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(args.learning_rate, warmup_steps=args.warmup_steps, t_total=t_total)
    else:
        scheduler = WarmupLinearSchedule(args.learning_rate, warmup_steps=args.warmup_steps, t_total=t_total)

    optimizer = paddle.optimizer.Momentum(parameters=model.parameters(),
                                learning_rate=scheduler,
                                momentum=0.9,
                                weight_decay=args.weight_decay,
                                grad_clip=paddle.nn.ClipGradByGlobalNorm(clip_norm=args.max_grad_norm))

    if args.resume_model is not None:
        start_iter = resume(args, model, optimizer, args.resume_model)
        scheduler.last_epoch = start_iter
    else:
        start_iter = 0

    if args.fp16:
        # Step1：定义 GradScaler，用于缩放loss比例，避免浮点数溢出
        scaler = paddle.amp.GradScaler(init_loss_scaling=args.loss_scale)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Total optimization steps = %d", args.num_steps)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * args.n_gpu)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)

    optimizer.clear_grad()
    set_seed(args.seed)  # Added here for reproducibility (even between python 2 and 3)
    losses = AverageMeter()
    global_step = start_iter
    best_acc = 0
    while True:
        model.train()
        
        for step, batch in enumerate(train_loader):
            x, y = batch
            if args.fp16:
                # Step2：创建AMP上下文环境，开启自动混合精度训练
                with paddle.amp.auto_cast():
                    loss = model(x, y)
            else:
                loss = model(x, y)

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            if args.fp16:
                # Step3：使用 Step1中定义的 GradScaler 完成 loss 的缩放，用缩放后的 loss 进行反向传播
                scaled = scaler.scale(loss)
                scaled.backward()
            else:
                loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                losses.update(loss.item()*args.gradient_accumulation_steps)
                if args.fp16:
                    # 训练模型
                    scaler.minimize(optimizer, scaled)
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.clear_grad()
                global_step += 1

                if global_step%10==0:
                    print("Training (%d / %d Steps) (loss=%2.5f(%2.5f))" % (global_step, t_total, losses.val, losses.avg))

                writer.add_scalar("train/loss", value=losses.val, step=global_step)
                writer.add_scalar("train/lr", value=optimizer.get_lr(), step=global_step)
                if global_step % args.eval_every == 0:
                    accuracy, valid_loss = valid(args, model, test_loader, global_step)
                    writer.add_scalar("test/accuracy", value=accuracy, step=global_step)
                    writer.add_scalar("test/loss", value=valid_loss, step=global_step)
                    print("Accuracy:",accuracy)
                    if best_acc < accuracy:
                        save_model(args, model, optimizer, global_step, is_best=True)
                        best_acc = accuracy
                    else:
                        save_model(args, model, optimizer, global_step, is_best=False)
                    model.train()

                if global_step % t_total == 0:
                    break
        losses.reset()
        if global_step % t_total == 0:
            break

    writer.close()
    logger.info("Best Accuracy: \t%f" % best_acc)
    logger.info("End Training!")
    print("Best Accuracy:",best_acc)


if __name__ == "__main__":
    
    # Model & Tokenizer Setup
    args, model = setup()

    # Training
    train_model(args, model)
