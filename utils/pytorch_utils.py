import os

def adjust_learning_rate(optimizer, epoch, initial_lr=0.001, decay_epoch=10):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = max(initial_lr * (0.1 ** (epoch // decay_epoch)), 1e-6)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class Save_Handle(object):
    """handle the number of """
    def __init__(self, max_num):
        self.save_list = []
        self.max_num = max_num

    def append(self, save_path):
        if len(self.save_list) < self.max_num:
            self.save_list.append(save_path)
        else:
            remove_path = self.save_list[0]
            del self.save_list[0]
            self.save_list.append(save_path)
            if os.path.exists(remove_path):
                os.remove(remove_path)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = 1.0 * self.sum / self.count

    def get_avg(self):
        return self.avg

    def get_count(self):
        return self.count


def set_trainable(model, requires_grad):
    for param in model.parameters():
        param.requires_grad = requires_grad



def get_num_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)