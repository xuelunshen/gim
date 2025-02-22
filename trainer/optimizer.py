import torch
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR, ExponentialLR
from torch.optim.lr_scheduler import _LRScheduler


def build_optimizer(model, config):
    name = config.TRAINER.OPTIMIZER
    lr = config.TRAINER.TRUE_LR

    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=config.TRAINER.ADAM_DECAY)
    elif name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=config.TRAINER.ADAMW_DECAY)
    else:
        raise ValueError(f"TRAINER.OPTIMIZER = {name} is not a valid optimizer!")


class PolynomialLRDecay(_LRScheduler):
    """Polynomial learning rate decay until step reach to max_decay_step

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        max_decay_steps: after this step, we stop decreasing learning rate
        end_learning_rate: scheduler stoping learning rate decay, value of learning rate must be this value
        power: The power of the polynomial.
    """

    def __init__(self, optimizer, max_decay_steps, end_learning_rate=0.0001, power=1.0):
        if max_decay_steps <= 1.:
            raise ValueError('max_decay_steps should be greater than 1.')
        self.max_decay_steps = max_decay_steps
        self.end_learning_rate = end_learning_rate
        self.power = power
        self.last_step = 0
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_step > self.max_decay_steps:
            return [self.end_learning_rate for _ in self.base_lrs]

        return [(base_lr - self.end_learning_rate) *
                ((1 - self.last_step / self.max_decay_steps) ** (self.power)) +
                self.end_learning_rate for base_lr in self.base_lrs]

    def step(self, step=None):
        if step is None:
            step = self.last_step + 1
        self.last_step = step if step != 0 else 1
        if self.last_step <= self.max_decay_steps:
            decay_lrs = [(base_lr - self.end_learning_rate) *
                         ((1 - self.last_step / self.max_decay_steps) ** (self.power)) +
                         self.end_learning_rate for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, decay_lrs):
                param_group['lr'] = lr


def build_scheduler(config, optimizer):
    """
    Returns:
        scheduler (dict):{
            'scheduler': lr_scheduler,
            'interval': 'step',  # or 'epoch'
            'monitor': 'val_f1', (optional)
            'frequency': x, (optional)
        }
    """
    scheduler = {'interval': config.TRAINER.SCHEDULER_INTERVAL}
    name = config.TRAINER.SCHEDULER

    if name == 'MultiStepLR':
        scheduler.update(
            {'scheduler': MultiStepLR(optimizer, config.TRAINER.MSLR_MILESTONES, gamma=config.TRAINER.MSLR_GAMMA)})
    elif name == 'CosineAnnealing':
        scheduler.update(
            {'scheduler': CosineAnnealingLR(optimizer, config.TRAINER.COSA_TMAX)})
    elif name == 'ExponentialLR':
        scheduler.update(
            {'scheduler': ExponentialLR(optimizer, config.TRAINER.ELR_GAMMA)})
    elif name == 'PolyLR':
        scheduler.update(
            {'scheduler': PolynomialLRDecay(optimizer, max_decay_steps=config.TRAINER.MAX_STEPS, end_learning_rate=0.0, power=1.0)})
    else:
        raise NotImplementedError()

    return scheduler


def get_lr_scheduler(optimizer, conf = None):
    """Get lr scheduler specified by conf.train.lr_schedule."""

    if conf is None:
        conf = {
            'type': 'exp',
            'start': 30,
            'exp_div_10': 10,
            'on_epoch': True,
            'factor': 1.0,
            'options': {}
        }

    if conf['type'] not in ["factor", "exp", None]:
        return getattr(torch.optim.lr_scheduler, conf['type'])(optimizer, **conf['options'])

    # backward compatibility
    def lr_fn(it):  # noqa: E306
        if conf['type'] is None:
            return 1
        if conf['type'] == "factor":
            return 1.0 if it < conf['start'] else conf['factor']
        if conf['type'] == "exp":
            gam = 10 ** (-1 / conf['exp_div_10'])
            return 1.0 if it < conf['start'] else gam
        else:
            raise ValueError(conf['type'])

    return torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_fn)
