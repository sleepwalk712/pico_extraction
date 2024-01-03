from app.models.db_models import Task, ModelVersion


def test_create_and_retrieve_task(db_setup):
    session = db_setup

    model_version = ModelVersion(
        ml_model_name="ml_model_name_example",
        version_number="1.0",
        path="/path/to/model",
    )

    task = Task(type="predict", status="pending", model_version=model_version)

    session.add(model_version)
    session.add(task)
    session.commit()

    retrieved_task = session.query(Task).first()

    assert retrieved_task is not None
    assert retrieved_task.type == "predict"
    assert retrieved_task.status == "pending"
    assert retrieved_task.model_version.ml_model_name == "ml_model_name_example"
