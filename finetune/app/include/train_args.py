# Number of worker threads for data loading (per RANK if Multi-GPU training).
NUM_WORKER_THREADS = 2

# Batch size used for training. This needs to be adjusted based on GPU VRAM and image size
BATCH_SIZE = 16

# Disables mosaic data augmentation in the last X epochs to stabilize training before completion. Setting to 0 disables this feature.
CLOSE_MOSAIC = 10

# Target image size for training. Images are resized to squares with sides equal to the specified value (if rect=False), preserving aspect ratio for YOLO models.
IMGZ_SHAPE = 1280  # 640  # Image shape: 2560x1489 too large, using 1280

# Initial learning rate. Adjusting this value is crucial for the optimization process, influencing how rapidly model weights are updated.
LEARNING_RATE = 0.001

# Randomly vary imgsz each batch by +/- multi_scale (e.g. 0.25 -> 0.75x to 1.25x), rounding to model stride multiples
MULTI_SCALE = 0

# Total number of training epochs.
NUM_EPOCHS = 100

# Choice of optimizer for training.
OPTIMIZER_NAME = "AdamW"

# Number of epochs to wait without improvement in validation metrics before early stopping the training.
PATIENCE = 20

# Enables minimum padding strategy; cannot use with multi-gpu training
RECT_FLAG = False

# Scales the image by a gain factor, simulating objects at different distances from the camera
# Default:0.5  This tells YOLO to zoom in significantly on images during training, effectively creating "crops" on the fly that keep the drone closer to its original size
SCALE = 0.8

# Number of epochs for learning rate warmup, gradually increasing the learning rate from a low value to the initial learning rate to stabilize training early on.
WARMUP_EPOCHS = 3
