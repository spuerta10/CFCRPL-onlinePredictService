from typing import Optional

from pydantic import BaseModel, model_validator


class MLModelConfSchema(BaseModel):
    """Class to parse and validate an MLmodel conf file. 
    """
    mlflow_ip: str
    mlmodel_name: str
    mlmodel_version: Optional[int] = None 
    mlmodel_alias: Optional[str] = None
    

    @model_validator(mode="before")
    @classmethod
    def check_version_or_alias(cls, values):
        """Check either mlmodel_version or mlmodel_alias is specified in the config file.

        Args:
        cls: The class that contains this validator. This is typically passed 
             automatically by Pydantic.
        values (dict): A dictionary of field values passed during model creation.

        Raises:
            ValueError: If neither 'mlmodel_version' nor 'mlmodel_alias' is provided.

        Returns:
            dict: The validated dictionary of values.
        """
        if not values.get("mlmodel_version") and not values.get("mlmodel_alias"):
            raise ValueError("Either 'mlmodel_version' or 'mlmodel_alias' must be provided.")
        return values