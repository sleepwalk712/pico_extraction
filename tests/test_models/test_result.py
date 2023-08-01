from app.models import Result, Task, ModelVersion


def test_create_and_retrieve_result(db_setup):
    session = db_setup

    task = Task(type="predict", status="pending")
    model_version = ModelVersion(
        model_name="model_name_example",
        version_number="1.0",
        path="/path/to/model",
    )

    session.add(task)
    session.add(model_version)
    session.commit()

    result = Result(
        task_id=task.task_id,
        data="some data",
        version_id=model_version.version_id,
    )
    session.add(result)
    session.commit()

    retrieved_result = session.query(Result).first()

    assert retrieved_result is not None
    assert retrieved_result.data == "some data"
    assert retrieved_result.task_id == task.task_id
    assert retrieved_result.version_id == model_version.version_id
