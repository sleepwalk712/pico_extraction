from pydantic import create_model
from typing import Optional


UserSchema = create_model(
    'UserSchema',
    user_id=(Optional[int], None),
    username=(str, ...),
    password_hash=(str, ...),
    role=(str, ...),
    __config__=dict(from_attributes=True, protected_namespaces=())
)
