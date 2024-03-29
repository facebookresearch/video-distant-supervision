# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Optimizer."""

import torch

import lib.utils.lr_policy as lr_policy


def construct_optimizer(model, cfg):
    """
    Construct a stochastic gradient descent or ADAM optimizer with momentum.
    Details can be found in:
    Herbert Robbins, and Sutton Monro. "A stochastic approximation method."
    and
    Diederik P.Kingma, and Jimmy Ba.
    "Adam: A Method for Stochastic Optimization."

    Args:
        model (model): model to perform stochastic gradient descent
        optimization or ADAM optimization.
        cfg (config): configs of hyper-parameters of SGD or ADAM, includes base
        learning rate,  momentum, weight_decay, dampening, and etc.
    """
    emb = []
    text = []
    if (cfg.TRAIN.SEP_LR  and not cfg.TRAIN.MULT == 1.) or cfg.TRAIN.LINEAR:
        
        # frozen parameters.
        bn_params = []
        # Non-frozen parameters.
        non_bn_parameters = []
        for name, p in model.named_parameters():
            if (not "head" in name and not 'text_lp' in name and not 'model.fc' in name) or (hasattr(cfg.MODEL, 'HEAD_T') and not cfg.MODEL.HEAD_T and "head."  in name):
                bn_params.append(p)
                if cfg.TRAIN.LINEAR: # and cfg.TRAIN.BATCH_SIZE > 8:
                    p.requires_grad = False
            elif 'head2' in name:
                emb.append(p)        
            else:
                non_bn_parameters.append(p)
                print(name)
        if cfg.TRAIN.LINEAR:     
            optim_params = [
                #{"params": bn_params, "weight_decay": 0., 'lr':0.},
                {"params": non_bn_parameters, "weight_decay": cfg.SOLVER.WEIGHT_DECAY, 'lr': cfg.SOLVER.BASE_LR, "lr_mult":1.},
            ]
        else:
            optim_params = [
            {"params": bn_params, "weight_decay": cfg.BN.WEIGHT_DECAY, "lr_mult":cfg.TRAIN.MULT},
            {"params": non_bn_parameters, "weight_decay": cfg.SOLVER.WEIGHT_DECAY, "lr_mult":1.},
        ]
        
        # Check all parameters will be passed into optimizer.
        assert len(list(model.parameters())) == len(non_bn_parameters) + len(
            bn_params) + len(emb)\
        , "parameter size does not match: {} + {} != {}".format(
            len(non_bn_parameters), len(bn_params), len(list(model.parameters()))
        )
    else:
        # Batchnorm parameters.
        bn_params = []
        bn_n=[]
        # Non-batchnorm parameters.
        non_bn_parameters = []
        non_bn_n=[]
        emb_n=[]
        text_n=[]
        for name, p in model.named_parameters():
            if "bn" in name:
                bn_params.append(p)
                bn_n.append(name)
            elif 'head2' in name:
                emb.append(p)
                emb_n.append(name)
            elif 'text_model' in name or 'text_module' in name:
                text.append(p)
                text_n.append(name)
                if cfg.TRAIN.MULT == 0:
                    p.requires_grad = False
            else:
                non_bn_parameters.append(p)
                non_bn_n.append(name)
        # Apply different weight decay to Batchnorm and non-batchnorm parameters.
        # In Caffe2 classification codebase the weight decay for batchnorm is 0.0.
        # Having a different weight decay on batchnorm might cause a performance
        # drop.
        print('bn\n',bn_n,'emb_n\n',emb_n,'no_bn\n',non_bn_n,'text\n',text_n)
        if cfg.TRAIN.MULT == 1:
            optim_params = [
            {"params": bn_params, "weight_decay": cfg.BN.WEIGHT_DECAY, "lr_mult":1.},
            {"params": non_bn_parameters+text, "weight_decay": cfg.SOLVER.WEIGHT_DECAY, "lr_mult":1.}
        ]
        elif cfg.TRAIN.MULT == 0:
            optim_params = [
            {"params": bn_params, "weight_decay": cfg.BN.WEIGHT_DECAY, "lr_mult":1.},
            {"params": non_bn_parameters, "weight_decay": cfg.SOLVER.WEIGHT_DECAY, "lr_mult":1.}
        ]
        else:    
            optim_params = [
            {"params": bn_params, "weight_decay": cfg.BN.WEIGHT_DECAY, "lr_mult":1.},
            {"params": non_bn_parameters, "weight_decay": cfg.SOLVER.WEIGHT_DECAY, "lr_mult":1.}, {"params": text, "weight_decay": cfg.SOLVER.WEIGHT_DECAY, "lr_mult":cfg.TRAIN.MULT}
        ]
        # Check all parameters will be passed into optimizer.
        assert len(list(model.parameters())) == len(non_bn_parameters) + len(
            bn_params) + len(emb)+ len(text)\
        , "parameter size does not match: {} + {} != {}".format(
            len(non_bn_parameters), len(bn_params), len(list(model.parameters()))
        )

    if cfg.SOLVER.OPTIMIZING_METHOD == "sgd":
        return torch.optim.SGD(
            optim_params,
            lr=cfg.SOLVER.BASE_LR,
            momentum=cfg.SOLVER.MOMENTUM,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            dampening=cfg.SOLVER.DAMPENING,
            nesterov=cfg.SOLVER.NESTEROV,
        )
    elif cfg.SOLVER.OPTIMIZING_METHOD == "adam":
        return torch.optim.Adam(
            optim_params,
            lr=cfg.SOLVER.BASE_LR,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        )
    elif cfg.SOLVER.OPTIMIZING_METHOD == "adamw":
        return torch.optim.AdamW(
            optim_params,
            lr=cfg.SOLVER.BASE_LR,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        )
    else:
        raise NotImplementedError(
            "Does not support {} optimizer".format(cfg.SOLVER.OPTIMIZING_METHOD)
        )


def get_epoch_lr(cur_epoch, cfg):
    """
    Retrieves the lr for the given epoch (as specified by the lr policy).
    Args:
        cfg (config): configs of hyper-parameters of ADAM, includes base
        learning rate, betas, and weight decays.
        cur_epoch (float): the number of epoch of the current training stage.
    """
    return lr_policy.get_lr_at_epoch(cfg, cur_epoch)


def set_lr(optimizer, new_lr):
    """
    Sets the optimizer lr to the specified value.
    Args:
        optimizer (optim): the optimizer using to optimize the current network.
        new_lr (float): the new learning rate to set.
    """
    for param_group in optimizer.param_groups:
        #if not param_group["lr"] == 0.:
            #print('new learning rate', new_lr * param_group["lr_mult"] if "lr_mult" in param_group else new_lr, len(param_group["params"]))
        param_group["lr"] = new_lr * param_group["lr_mult"] if "lr_mult" in param_group else new_lr
