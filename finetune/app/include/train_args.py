NUM_WORKER_THREADS = 2  # 4

BATCH_SIZE = 16  # 8  # 16
CLOSE_MOSAIC = 10
IMGZ_SHAPE = 1280  # 1024  # 640  #Image shape: 2560x1489 too large, using 1280
LEARNING_RATE = 0.001  # 0.001
MULTI_SCALE = 0  # .75  #True  # Change imgsz by up to a factor of 0.5 during training to be more accurate with multiple imgsz during inference
NUM_EPOCHS = 100  # 60
OPTIMIZER_NAME = "AdamW"
PATIENCE = 20  # 5  # Automatically stops training if no improvement after P epochs [Default: 100]
RECT_FLAG = False  # True  # Enables minimum padding strategy; cannot use with multi-gpu training
SCALE = 0.8  # Default:0.5  This tells YOLO to zoom in significantly on your 2560px images during training, effectively creating "crops" on the fly that keep the drone closer to its original size
WARMUP_EPOCHS = (
    3  # Set to 0 to prevent the learning rate from starting too low [Default: 3]
)
