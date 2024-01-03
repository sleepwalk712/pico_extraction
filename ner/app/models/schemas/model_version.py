from pydantic import create_model
from datetime import datetime
from typing import Optional


ModelVersionSchema = create_model(
    'ModelVersionSchema',
    version_id=(Optional[int], None),
    ml_model_name=(str, ...),
    version_number=(str, ...),
    path=(str, ...),
    created_at=(Optional[datetime], None),
    __config__=dict(from_attributes=True)
)
