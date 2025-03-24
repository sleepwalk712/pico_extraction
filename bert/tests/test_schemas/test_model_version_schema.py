from datetime import datetime

from app.models.schemas import ModelVersionSchema


def test_model_version_schema():
    test_data = {
        "ml_model_name": "ml_model_name_example",
        "version_number": "1.0",
        "path": "/path/to/model"
    }

    model_version = ModelVersionSchema(**test_data)

    assert model_version.ml_model_name == "ml_model_name_example"
    assert model_version.version_number == "1.0"
    assert model_version.path == "/path/to/model"

    assert model_version.created_at is None or isinstance(
        model_version.created_at, datetime)

    orm_data = {
        "version_id": 1,
        "ml_model_name": "ml_model_name_example",
        "version_number": "1.0",
        "path": "/path/to/model",
        "created_at": datetime.now()
    }

    orm_version = ModelVersionSchema.model_validate(orm_data)
    assert orm_version.version_id == 1
    assert orm_version.ml_model_name == "ml_model_name_example"
    assert orm_version.version_number == "1.0"
    assert orm_version.path == "/path/to/model"
    assert isinstance(orm_version.created_at, datetime)
