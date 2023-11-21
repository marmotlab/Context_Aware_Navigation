FOLDER_NAME = 'Test'
model_path = f'model/{FOLDER_NAME}'
gifs_path = f'results/{FOLDER_NAME}/gifs'
trajectory_path = f'results/trajectory'
length_path = f'results/length'
INPUT_DIM = 7
EMBEDDING_DIM = 128
K_SIZE = 20  # the number of neighbors
USE_GPU = False  # do you want to use GPUS?
NUM_GPU = 1 # the number of GPUs
NUM_META_AGENT = 1  # the number of processes
NUM_TEST = 100
NUM_RUN = 1
SAVE_GIFS = True  # do you want to save GIFs
SAVE_TRAJECTORY = True  # do you want to save per-step metrics
SAVE_LENGTH = True  # do you want to save per-episode metrics