
class Config():
    seed = 42

    # path
    datafolder = 'path/'
    experiment = 'MITSupra'
    model_name = 'MYMODNet'

    lamda_a2b = 0.1
    lamda_cof = 0.005

    batch_size = 64

    n_segments1 = 59
    n_segments2 = 8

    max_epoch = 100

    lr = 0.001

    device_num = 1


config = Config()
