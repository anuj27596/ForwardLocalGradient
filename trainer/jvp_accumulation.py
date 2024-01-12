import os
import torch
from functools import partial
from tqdm import tqdm

from model.utils import init_weights
from trainer.gradients import stash_auxiliary_gradients, restore_auxiliary_gradients, \
    get_gradient_computation_function
from trainer.accumulators import accumulate_jvp
from trainer.utils import AverageMeter, set_bn_train, set_bn_eval
# from trainer.hooks import set_net_hook
from trainer.trackers import reset_tracker


def train_epoch(dataloader, net, criterion, optimizer, target='global', guess='local-and-last', space='weight', device=None, args=None):
    # guess
    compute_guess = partial(get_gradient_computation_function(guess), dest='guess', space=space)
    if args.training.guess == 'random':
        compute_guess = partial(compute_guess, noise_type=args.training.noise_type)

    # tracker
    net.train()
    net.to(device)
    criterion.to(device)
    reset_tracker(net)

    for block in net.blocks:
        block.auxnet.loss = AverageMeter()
        block.auxnet.accs = AverageMeter()
    
    for data, target in tqdm(dataloader):
        data, target = data.to(device), target.to(device)

        # guess gradient
        optimizer.zero_grad()
        
        # handles = set_net_hook(net, space, 'guess') # anuj: may need for per-sample projection

        compute_guess(net, data, target, criterion)

        stash_auxiliary_gradients(net)
        
        # for h in handles:
        #     h.remove()

        # compute projection
        with torch.no_grad():
            accumulate_jvp(net, data, target, criterion)

        restore_auxiliary_gradients(net)

        optimizer.step()

    return net.blocks[-1].auxnet.loss.avg, net.blocks[-1].auxnet.accs.avg

