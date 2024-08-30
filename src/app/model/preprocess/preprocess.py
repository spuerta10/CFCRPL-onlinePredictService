from credit_risk_pipeline_lib.pipeline.ingest.json import JSON
from credit_risk_pipeline_lib.pipeline.transform.transform_interface import TransformInterface 
from credit_risk_pipeline_lib.pipeline.pipeline import Pipeline

from pandas import DataFrame, concat
from sklearn.preprocessing import LabelEncoder


class Transform(TransformInterface):
    def __init__(self, step_name: str = "Sklearn encode cat vars"):
        self._step_name = step_name
    
    
    def transform(self, data: DataFrame) -> DataFrame:
        data = data.convert_dtypes()
        encoded_cols: DataFrame = data.select_dtypes(include=["string"]) \
            .apply(lambda x: LabelEncoder().fit_transform(x))
        encoded_cols.columns = [f"encoded_{col}" for col in encoded_cols.columns]
        transformed_data = concat([data, encoded_cols], axis =1)
        return transformed_data.drop(data.select_dtypes(include=["string"]), axis=1)

       
class Preprocess:
    @staticmethod
    def run(data: str | dict):
        ingest = JSON(data)
        transform = Transform()
        preprocessed_data: DataFrame = Pipeline(
            ingest_stage= ingest,
            transform_stage= transform
        ).run()
        return preprocessed_data
        
        