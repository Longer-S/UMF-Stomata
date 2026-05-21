
class args():

    # training args
    epochs = 50
    batch_size = 16
    trainDataset = 'datasets/train/stack/'
    valDataset = 'datasets/val/stack/'
    trainNumber = 21780

    n_stack = 21
    in_channels = 3
    out_channels = 1
    HEIGHT = 256
    WIDTH = 256
    

    save_model_dir = "multi-focus/models" 
    save_loss_dir = "multi-focus/models/loss"  

    cuda = 1 
    seed = 2025

    lr = 1e-4
    use_scheduler=False
    scheduler_type='step'
    lr_step_size=200
    lr_gamma=0.5

    log_interval = 2 
    resume =None
    device = 0

    # testing args
    PATCH_SIZE = 128  
    model_path = "best.pth"




