# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import time

import torch
import torch.distributed as dist
from maskrcnn_benchmark.utils.comm import get_world_size
from maskrcnn_benchmark.utils.metric_logger import MetricLogger
from tqdm import tqdm


def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def do_train(
        model,  # model object created by build_detection_model() function
        data_loader,  # dataset
        optimizer,  # object for torch.optim.sgd.SGD
        scheduler,  # learning rate updating strategy
        checkpointer,
        device,  # torch.device: used to decide hardware training device
        checkpoint_period,  # model weight saving period
        arguments,  # extra parameters, e.g. arguments[iteration]
        summary_writer=None
):
    # record log information
    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    logger.info("Start training")

    # used to record 
    meters = MetricLogger(delimiter="  ")
    max_iter = len(data_loader)
    start_iter = arguments["iteration"]
    model.train()  # set the model in training mode
    start_training_time = time.time()
    end = time.time()

    # with torch.profiler.profile(
    #         schedule=torch.profiler.schedule(
    #             wait=4,
    #             warmup=2,
    #             active=6,
    #             repeat=1),
    #         on_trace_ready=torch.profiler.tensorboard_trace_handler('logs/profilerIS4'),
    #         with_stack=True
    # ) as profiler:

    for iteration, (images, targets, proposals, _) in enumerate(data_loader, start_iter):
        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration

        images = images.to(device)
        targets = [target.to(device) for target in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict[0].values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict[0])
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)

        optimizer.zero_grad()
        # Note: If mixed precision is not used, this ends up doing nothing
        # Otherwise apply loss scaling for mixed-precision recipe
        losses.backward()
        optimizer.step()
        scheduler.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 100 == 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )
            if summary_writer:
                loss_global_avg = meters.loss.global_avg
                loss_median = meters.loss.median
                loss_mask = meters.loss_mask.global_avg
                # print('loss global average: {0}, loss median: {1}'.format(meters.loss.global_avg, meters.loss.median))
                summary_writer.add_scalar('train_loss_global_avg', loss_global_avg, iteration)
                summary_writer.add_scalar('train_loss_median', loss_median, iteration)
                summary_writer.add_scalar('train_loss_raw', losses_reduced, iteration)
                summary_writer.add_scalar('train_loss_mask', loss_mask, iteration)
        if iteration % checkpoint_period == 0:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)
        if iteration == max_iter:
            checkpointer.save("model_final", **arguments)

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )
