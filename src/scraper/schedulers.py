from transformers import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup
import torch
import math
import pytorch_warmup as warmup

def compute_exp_lr_with_warmup(epoch, warmup_steps, gamma):
    if epoch < warmup_steps:
         return float(epoch) / float(max(1, warmup_steps))
    return gamma ** epoch

class LearningRateScheduler():
    """Factory for LR Schedulers. It abstracts the step function so that we can use pytorch_warmup package"""
    def __init__(self, optimizer, num_warmup_steps, num_training_steps, scheduler_type, **kwargs):
        self.scheduler_type = scheduler_type
        self.scheduler, self.warmup_scheduler = self.create_scheduler(optimizer, num_warmup_steps, num_training_steps, scheduler_type, **kwargs)
    
    def create_scheduler(self, optimizer, num_warmup_steps, num_training_steps, scheduler_type="linear_with_warmup", **kwargs):
        if scheduler_type == "linear_with_warmup":
            return get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps), None
        if scheduler_type == "constant_with_warmup":
            return get_constant_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps), None
        if scheduler_type == "cosine_with_warmup":
            return get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps), None
        if scheduler_type == "cyclic_triangular2":
            return torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-8, max_lr=kwargs['lr'],step_size_up=kwargs['epoch_steps'] // 2,mode="triangular2"), None
        if scheduler_type == "cyclic_exp":
            return torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1-8, max_lr=kwargs['lr'],step_size_up=kwargs['epoch_steps'] // 2,mode="exp_range",gamma=0.85), None
        if scheduler_type == "step_with_warmup":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=kwargs['epoch_steps'], gamma=0.1)
            warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period=num_warmup_steps)
            return scheduler, warmup_scheduler
        if scheduler_type == "exponential_with_warmup":
            base_lr = 1e-8
            max_lr = kwargs['lr']
            steps = kwargs['epoch_steps']
            gamma = math.e ** ((1 / steps) * (math.log(base_lr) - math.log(max_lr)))
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: compute_exp_lr_with_warmup(epoch, num_warmup_steps, gamma))
            return scheduler, None
    
    def get_scheduler(self):
        return self.scheduler

    def step(self):
        if self.scheduler_type in ['step_with_warmup']:
            with self.warmup_scheduler.dampening():
                self.scheduler.step()      
        else:
            self.scheduler.step()
