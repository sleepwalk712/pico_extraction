from datetime import datetime

from app.models.schemas import ResultSchema


def test_result_schema():
    test_data = {
        "task_id": 123,
        "data": "Test data"
    }

    result = ResultSchema(**test_data)

    assert result.task_id == 123
    assert result.data == "Test data"
    assert result.created_at is None

    orm_data = {
        "result_id": 1,
        "task_id": 123,
        "data": "Test data",
        "created_at": datetime.now()
    }

    orm_version = ResultSchema.model_validate(orm_data)

    assert orm_version.result_id == 1
    assert orm_version.task_id == 123
    assert orm_version.data == "Test data"
    assert isinstance(orm_version.created_at, datetime)
