import torch
import torch.autograd.forward_ad as fwAD


# ========== Jacobian-vector product Accumulator ==========

def dualize(module):
    setattr(module, 'param_stash', {})
    for name, param in list(module.named_parameters(recurse = False)):
        module.param_stash[name] = param
        delattr(module, name)
        setattr(module, name, fwAD.make_dual(param.detach().clone(), param.grad.clone()))


def undualize(module):
    for name, param in module.param_stash.items():
        delattr(module, name)
        # import ipdb;ipdb.set_trace()
        module.register_parameter(name, param)
    delattr(module, 'param_stash')


def accumulate_jvp(net, data, target, criterion):
    with fwAD.dual_level():
        net.apply(dualize)
        pred = net(data)
        loss = criterion(pred, target)
        jvp = fwAD.unpack_dual(loss).tangent

        net.apply(undualize)

    jvp = jvp / sum([(param.grad ** 2).sum() for param in net.parameters()])
    # jvp = (jvp * 1e4).clip(-1, 1)

    for param in net.parameters():
        param.grad *= jvp

