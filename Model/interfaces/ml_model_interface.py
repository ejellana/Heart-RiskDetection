# Model/interfaces/ml_model_interface.py
from abc import ABC, abstractmethod

class MLModelInterface(ABC):
    @abstractmethod
    def Predict(self, features):
        pass