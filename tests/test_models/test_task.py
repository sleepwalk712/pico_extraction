from app.models import Task


def test_create_and_retrieve_task(db_setup):
    session = db_setup
    task = Task(type="predict", status="pending")
    session.add(task)
    session.commit()

    retrieved_task = session.query(Task).first()

    assert retrieved_task is not None
    assert retrieved_task.type == "predict"
    assert retrieved_task.status == "pending"
