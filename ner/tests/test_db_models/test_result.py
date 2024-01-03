from app.models.db_models import Result, Task, ModelVersion


def test_create_and_retrieve_result(db_setup):
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

    result = Result(
        task_id=task.task_id,
        data="some data",
    )
    session.add(result)
    session.commit()

    retrieved_result = session.query(Result).first()

    assert retrieved_result is not None
    assert retrieved_result.data == "some data"
    assert retrieved_result.task_id == task.task_id
    assert retrieved_result.task.model_version.version_id == model_version.version_id
