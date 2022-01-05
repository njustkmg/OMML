import paddle


class ReduceLROnPlateau(object):
    "Optim wrapper that implements rate."
    def __init__(self, init_lr, model, grad_clip, mode='min', factor=0.1, patience=10, verbose=False, threshold=0.0001,
                 threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-8):
        self.scheduler = paddle.optimizer.lr.ReduceOnPlateau(init_lr, mode, factor, patience, threshold, threshold_mode, cooldown, min_lr, eps, verbose)
        self.optimizer = paddle.optimizer.Adam(learning_rate=self.scheduler,
                                               parameters=model.parameters(),
                                               beta1=0.9, beta2=0.999,
                                               epsilon=1e-8,
                                               grad_clip=paddle.fluid.clip.ClipGradByValue(grad_clip))
        self.current_lr = self.optimizer.get_lr()

    def step(self):
        "Update parameters and rate"
        self.optimizer.step()

    def clear_grad(self):
        """Clear the grad."""
        self.optimizer.clear_grad()

    def scheduler_step(self, val):
        self.scheduler.step(val)
        self.current_lr = self.optimizer.get_lr()

    def state_dict(self):
        return {'current_lr':self.current_lr,
                'scheduler_state_dict': self.scheduler.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict()}

    def load_state_dict(self, state_dict):
        if 'current_lr' not in state_dict:
            # it's normal optimizer
            self.optimizer.set_state_dict(state_dict)
            self.optimizer.set_lr(self.current_lr)
        else:
            # it's a schduler
            self.current_lr = state_dict['current_lr']
            self.scheduler.set_state_dict(state_dict['scheduler_state_dict'])
            self.optimizer.set_state_dict(state_dict['optimizer_state_dict'])
