from app.models.db_models import User


def test_create_and_retrieve_user(db_setup):
    session = db_setup

    user = User(
        username="test_username",
        password_hash="hashed_password",
        role="user_role",
    )
    session.add(user)
    session.commit()

    retrieved_user = session.query(User).first()

    assert retrieved_user is not None
    assert retrieved_user.username == "test_username"
    assert retrieved_user.password_hash == "hashed_password"
    assert retrieved_user.role == "user_role"
