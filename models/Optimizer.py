from torch.optim import Optimizer, SGD

class SCAFFOLDOptimizer(SGD):
    def __init__(self, params, lr, momentum=0, dampening=0, weight_decay=0, nesterov=False):
        super(SCAFFOLDOptimizer, self).__init__(params, lr, momentum, dampening, weight_decay, nesterov)

    def step(self, client_cs, server_cs):
        for group in self.param_groups:
            for p, sc, cc in zip(group['params'], 
                                 (v for k,v in server_cs.items() if 'running' not in k and 'track' not in k), 
                                 (v for k,v in client_cs.items() if 'running' not in k and 'track' not in k)):# state_dict 转 named_parameter
                if p.grad is not None:
                    p.grad.data = p.grad.data + sc - cc
        super(SCAFFOLDOptimizer, self).step()

        # for group in self.param_groups:
        #     for p, sc, cc in zip(group['params'], server_cs, client_cs):
        #         p.data.add_(other=(p.grad.data + sc - cc), alpha=-group['lr'])