from abc import ABC, abstractmethod

class PreprocessInterface(ABC):
    @abstractmethod
    def run(data): 
        pass