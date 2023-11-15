from app.models.schemas import UserSchema


def test_user_schema():
    test_data = {
        "username": "test_user",
        "password_hash": "hashed_password",
        "role": "user"
    }

    user = UserSchema(**test_data)

    assert user.username == "test_user"
    assert user.password_hash == "hashed_password"
    assert user.role == "user"
    assert user.user_id is None

    orm_data = {
        "user_id": 1,
        "username": "test_user",
        "password_hash": "hashed_password",
        "role": "user"
    }

    orm_version = UserSchema.model_validate(orm_data)

    assert orm_version.user_id == 1
    assert orm_version.username == "test_user"
    assert orm_version.password_hash == "hashed_password"
    assert orm_version.role == "user"
