from app.controller.request_parser import RequestParser
from app.model.model import Model
from app.model.preprocess.preprocess import Preprocess 

from flask import (
    Blueprint,
    request,
    jsonify
)


controller_bp = Blueprint("controller", __name__)


@controller_bp.route("/predict", methods=["POST"])
def predict():
    """Parses, preprocess and asks a prediction from a MLModel based on clients requests data.

    Raises:
        ValueError: If a clients request is empty.

    Returns:
        _type_: _description_
    """
    try:
        request_json: dict = request.get_json(silent=True)
        if request_json is None: 
            raise ValueError("The given request is empty!")
        RequestParser(**request_json)
        prediction = (
            Model(request_json)
            .preprocess_for_mlmodel(Preprocess)
            .get_prediction()
        )
        return jsonify(
            {"prediction":str(prediction)}
        ), 200
    except Exception as e:
        return jsonify({"Exception!": str(e)}), 400
        