from app.model.preprocess.preprocess_interface import PreprocessInterface

from pandas import DataFrame
from numpy import ndarray
from credit_risk_pipeline_lib.mlmodel.mlmodel import MLModel
from app.conf.mlmodel_conf_schema import MLModelConfSchema 

class Model:
    def __init__(self, data: dict):
        self._data = data
        self._preprocessed_data = None
        
        
    def preprocess_for_mlmodel(self, preprocess_pipeline: PreprocessInterface) -> 'Model':
        """Transforms data running given preprocess pipeline.

        Args:
            preprocess_pipeline (PreprocessInterface): Pipeline and steps to run to preprocess data (transform data to a format accepted by MLmodel). 

        Returns:
            Model: Object with an attribute storing data passed to Preprocess pipeline (preprocessed data).
        """
        self._preprocessed_data: DataFrame = preprocess_pipeline.run(self._data)
        return self
    
    
    def get_prediction(self) -> ndarray: 
        """Asks MlModel for a prediction based on preprocessed data. 

        Returns:
            ndarray: Prediction of the model based on given data.
        """
        if self._preprocessed_data is None:
            raise ValueError("Run preprocess_for_mlmodel first!")
        prediction: ndarray = (
            MLModel(r"src/app/conf/xgb_conf.json", MLModelConfSchema)
            .fetch("mlmodel_name", "mlmodel_version")
            .predict(self._preprocessed_data)
        )
        return prediction