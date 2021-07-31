from easydict import EasyDict as edict


class Config:
    # dataset
    DATASET = edict()
    DATASET.DATASETS = ['DIK2K', 'Flickr2K']
    DATASET.SCALE = 2
    DATASET.PATCH_WIDTH = 32
    DATASET.PATCH_HEIGHT = 32
    DATASET.REPEAT = 1
    DATASET.AUG_MODE = True
    DATASET.VALUE_RANGE = 255.0
    DATASET.SEED = 0

    # dataloader
    DATALOADER = edict()
    DATALOADER.IMG_PER_GPU = 1024
    DATALOADER.NUM_WORKERS = 4

    # model
    MODEL = edict()
    MODEL.KERNEL_SIZE = 5
    MODEL.NWEIGT = 21
    # MODEL.KERNEL_PATH = '/home/redrock/project/sisr_project_0405/model/models/baseline_lwb_v1/quadrant_k5.pkl'
    MODEL.KERNEL_PATH = '//home/redrock/project/sisr_drn/adakernel/kernels/g5_step6_dog/filters.npy'
    MODEL.fineturningK = False

    MODEL.angleDims = 24 
    MODEL.coDims = 3 
    MODEL.angle_blocks = 16 
    MODEL.co_blocks = 16 
    MODEL.weight_act = None

    MODEL.IN_CHANNEL = 3

    MODEL.N_CHANNEL = 32
    MODEL.OUT_CHANNELS = [24, 3]
    MODEL.N_BLOCK = 2
    MODEL.DOWN = 4
    MODEL.DEVICE = 'cuda'

    MODEL.SCALE = 1
    MODEL.RES_BLOCK = 3
    MODEL.NFEAT = 32
    MODEL.block_feats = 32

    # solver
    SOLVER = edict()
    SOLVER.OPTIMIZER = 'Adam'
    SOLVER.BASE_LR = 1e-4
    SOLVER.WARM_UP_FACTOR = 0.1
    SOLVER.WARM_UP_ITER = 2000
    SOLVER.MAX_ITER = 200000
    SOLVER.WEIGHT_DECAY = 0
    SOLVER.MOMENTUM = 0
    SOLVER.BIAS_WEIGHT = 0.0

    # initialization
    CONTINUE_ITER = None
    INIT_MODEL = None #'/data1/zk/vsr/sisir_project_0405/model/models/baseline_lwb_v1/200000.pth'

    # log and save
    LOG_PERIOD = 20
    SAVE_PERIOD = 5000

    # validation
    VAL = edict()
    VAL.PERIOD = 5000
    VAL.DATASET = 'Set5'
    VAL.IMG_PER_GPU = 1
    VAL.NUM_WORKERS = 1
    VAL.MAX_NUM = 100
    VAL.SAVE_IMG = True
    VAL.TO_Y = True


config = Config()



