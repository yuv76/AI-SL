import os

NUM_CLASSES = 23  # all letters that are not AEMNST
NUM_COMBINED_CLASSES = 6  # AEMNST

SUB_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'final_submodel_used_diff_dataset.pth')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model_second_try.pth')
