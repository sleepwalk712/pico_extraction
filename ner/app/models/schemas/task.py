from pydantic import create_model
from datetime import datetime
from typing import Optional


TaskSchema = create_model(
    'TaskSchema',
    task_id=(Optional[int], None),
    type=(str, ...),
    status=(str, ...),
    created_at=(Optional[datetime], None),
    completed_at=(Optional[datetime], None),
    description=(Optional[str], None),
    version_id=(Optional[int], None),
    __config__=dict(from_attributes=True, protected_namespaces=())
)
