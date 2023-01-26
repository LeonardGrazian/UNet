
import os

# data settings
DATA_DIR = '/Users/leonard/Workspace/ML/CV/UNet/data/'
IMAGE_DIR = os.path.join(DATA_DIR, 'oxford-iiit-pet/images/')
MASK_DIR = os.path.join(DATA_DIR, 'oxford-iiit-pet/annotations/trimaps/')
MODEL_FILENAME = 'saved_models/unet.pth'
VALIDATION_FRACTION = 0.1

# training hyperparameters
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-4

# other constants
INSPECT = False # inspect output of trained model
N_INSPECT_IMAGES = 10 # number of images to inspect
