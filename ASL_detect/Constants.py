INPUT_SIZE = 64*64  # 64*64 pixels (not directly used in CNN)
NUM_CLASSES = 23  # all letters
NUM_COMBINED_CLASSES = 6  # AEMNST
LEARNING_RATE = 0.001
BATCH_SIZE = 128
NUM_EPOCHS = 100
TRAIN_RATIO = 0.7
PATIENCE = 3
TEST_DATA_PATH = "Dataset\\mydata\\test_set\\"
TRAIN_DATA_PATH = "Dataset\\mydata\\training_set\\"
PATH_MODEL = 'finished_model.pth'

PATH_CHECKPOINT = 'model_checkpoint.pth'

SUB_MODEL_PATH = 'final_submodel_used_diff_dataset.pth'
MODEL_PATH = 'model_second_try.pth'

