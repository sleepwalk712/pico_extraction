from datetime import datetime

from app.models.schemas import TaskSchema


def test_task_schema():
    test_data = {
        "type": "Example Type",
        "status": "Example Status",
        "description": "Example Description"
    }

    task = TaskSchema(**test_data)

    assert task.type == "Example Type"
    assert task.status == "Example Status"
    assert task.description == "Example Description"
    assert task.created_at is None
    assert task.completed_at is None

    orm_data = {
        "task_id": 1,
        "type": "Example Type",
        "status": "Example Status",
        "created_at": datetime.now(),
        "completed_at": datetime.now(),
        "description": "Example Description",
        "version_id": 123
    }

    orm_version = TaskSchema.model_validate(orm_data)

    assert orm_version.task_id == 1
    assert orm_version.type == "Example Type"
    assert orm_version.status == "Example Status"
    assert isinstance(orm_version.created_at, datetime)
    assert isinstance(orm_version.completed_at, datetime)
    assert orm_version.description == "Example Description"
    assert orm_version.version_id == 123
