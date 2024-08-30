from app.model.preprocess.preprocess_interface import PreprocessInterface

from pandas import DataFrame
from numpy import ndarray
from credit_risk_pipeline_lib.mlmodel.mlmodel import MlModel
from app.conf.conf_schema import ConfSchema

class Model:
    def __init__(self, data: dict):
        self._data = data
        self._preprocessed_data = None
        
        
    def preprocess_for_mlmodel(self, preprocess_pipeline: PreprocessInterface) -> 'Model':
        self._preprocessed_data: DataFrame = preprocess_pipeline.run(self._data)
        return self
    
    
    def get_prediction(self) -> ndarray: 
        if self._preprocessed_data is None:
            raise ValueError("Run preprocess_for_mlmodel first!")
        prediction: ndarray = (
            MlModel(r"src/app/conf/xgb_conf.json", ConfSchema)
            .fetch("mlmodel_name", "mlmodel_version")
            .predict(self._preprocessed_data)
        )
        return prediction