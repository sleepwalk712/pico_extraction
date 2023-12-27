from fastapi import FastAPI

from app.api.v1 import pico


app = FastAPI()

app.include_router(pico.router, prefix="/v1/pico")


@app.get("/")
def read_root() -> dict[str, str]:
    return {"Hello": "World"}
