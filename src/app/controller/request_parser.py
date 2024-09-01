from pydantic import BaseModel, field_validator


class RequestParser(BaseModel):
    """Class to parse and validate any clients requests. 
    """
    person_age: int
    person_income: float
    person_home_ownership: str
    person_emp_length: float
    loan_intent: str
    loan_grade: str
    loan_amnt: float
    loan_int_rate: float
    loan_percent_income: float 
    cb_person_default_on_file: str
    cb_person_cred_hist_length: int 
    
    
    @field_validator("loan_grade")
    @classmethod
    def check_single_letter(cls, field_value: str, field_name: str):
        """Checks if a given field value contains an upper case letter 
        raging from A-Z.

        Args:
            field_value (str): The value of the field that's been checked.
            field_name (str): The name of the field that's been checked.
        
        Raises:
            ValueError: If the value checking process fails.

        Returns:
            str: The checked fields value.
        """
        if not field_value.isupper() or not field_value.isalpha() or len(field_value) != 1:
            error = f"""
            Field {field_name} was expecting upper case letter raging from A-Z, 
            got {field_value}
            """
            raise ValueError(error)
        return field_value
    
    
    @field_validator("cb_person_default_on_file")
    @classmethod
    def check_y_or_n(cls, field_value: str, field_name: str):
        """Checks if a given field value is either a Y or a N. 

        Args:
            field_value (str): The value of the field that's been checked.
            field_name (str): The name of the field that's been checked.

        Raises:
            ValueError: If the value checking process fails.

        Returns:
            str: The checked fields value.  
        """
        if field_value not in {"Y", "N"}:
            error = f"""
            Field {field_name} was expecting Y or N, 
            got {field_value}
            """
            raise ValueError(error)
        return field_value


    @field_validator("person_home_ownership", "loan_intent")
    @classmethod
    def check_uppercase(cls, field_value: str, field_name: str):
        """_summary_

        Args:
            field_value (str): The value of the field that's been checked.
            field_name (str): The name of the field that's been checked.
        
        Raises:
            ValueError: If the value checking process fails. 

        Returns:
            str: The checked fields value.      
        """
        if not field_value.isupper():
            error = f"""
            Field {field_name} was expecting {field_value.upper()}, 
            got {field_value}
            """
            raise ValueError(error)
        return field_value