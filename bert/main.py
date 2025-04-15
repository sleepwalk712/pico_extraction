from fastapi import FastAPI

from app.api.v1 import ner
from app.api.v1 import classification


app = FastAPI()

app.include_router(ner.router, prefix="/v1")
app.include_router(classification.router, prefix="/v1")


@app.get("/")
def read_root() -> dict[str, str]:
    return {"Hello": "World"}
