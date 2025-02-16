import os
import ASL_detect.ModelPytorchCustomLayers as ModelPytorch
from ASL_detect.Constants import MODEL_PATH, SUB_MODEL_PATH, NUM_CLASSES, NUM_COMBINED_CLASSES
import torch

class loadModel:
    model = None
    sub_model = None

    @classmethod
    def get_model(cls):
        if cls.model is None:
            cls.model = loadModel.load_model(MODEL_PATH, NUM_CLASSES)
        return cls.model

    @classmethod
    def get_sub_model(cls):
        if cls.sub_model is None:
            cls.sub_model = loadModel.load_model(SUB_MODEL_PATH, NUM_COMBINED_CLASSES)
        return cls.sub_model

    @classmethod
    def load_model(cls, model_path, num_classes):
        cnn_model = ModelPytorch.CNN(in_channels=1, num_classes=num_classes)
        # Load the saved state dictionary
        if os.path.exists(model_path):
            cnn_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        else:
            raise FileNotFoundError(f"Model file not found at {model_path}")
        cnn_model.eval()
        return cnn_model
