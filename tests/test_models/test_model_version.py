import datetime

from app.models import ModelVersion


def test_create_and_retrieve_model_version(db_setup):
    session = db_setup

    model_version = ModelVersion(
        model_name="model_name_example",
        version_number="1.0",
        path="/path/to/model",
    )
    session.add(model_version)
    session.commit()

    retrieved_model_version = session.query(ModelVersion).first()

    assert retrieved_model_version is not None
    assert retrieved_model_version.model_name == "model_name_example"
    assert retrieved_model_version.version_number == "1.0"
    assert retrieved_model_version.path == "/path/to/model"
    assert isinstance(retrieved_model_version.created_at, datetime.datetime)
