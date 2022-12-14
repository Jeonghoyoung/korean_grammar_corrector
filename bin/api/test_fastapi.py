import time
import uvicorn
from fastapi import FastAPI, Form

app = FastAPI()

@app.get('/')
def root():
    return {'message': 'Hello World'}


@app.get('/items/{item_id}')
def read_item(item_id:int, q:str=None):
    return {"item_id": item_id, 'q':q}

@app.post("/login/")
async def login(username: str = Form(), password: str = Form()):
    return {"username": username}

# if __name__ == '__main__':
    # uvicorn test_fastapi:app --reload --host=0.0.0.0 --port=8000
