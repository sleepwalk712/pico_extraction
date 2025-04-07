from pydantic import create_model
from datetime import datetime
from typing import Optional


ResultSchema = create_model(
    'ResultSchema',
    result_id=(Optional[int], None),
    task_id=(int, ...),
    data=(str, ...),
    created_at=(Optional[datetime], None),
    __config__=dict(from_attributes=True, protected_namespaces=())
)
