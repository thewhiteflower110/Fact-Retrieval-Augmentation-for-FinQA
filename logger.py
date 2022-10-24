from torch.utils.tensorboard import SummaryWriter

#file to write all the logs

# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter('runs/fashion_mnist_experiment_1')

def train_log(dict):
    writer.add_scalar('training loss',dict["loss"])
    writer.add_scalar("Loss/train", loss, epoch)

def val_log(dict):

writer.add_figure('validation loss',)
