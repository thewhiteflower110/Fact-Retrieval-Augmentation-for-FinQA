from torch.utils.tensorboard import SummaryWriter

#file to write all the logs

# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter('runs/fashion_mnist_experiment_1')

def train_log(loss, accuracy):
    for l, a, e in zip(loss, accuracy, range(len(loss))):
        writer.add_scalar("Training/loss", l, e)
        writer.add_scalar("Training/accuracy", a, e)
        
def val_log(loss, accuracy):
    for l, a, e in zip(loss, accuracy, range(len(loss))):
        writer.add_scalar("Validation/loss", l, e)
        writer.add_scalar("Validation/accuracy", a, e)

writer.add_figure('validation loss',)
