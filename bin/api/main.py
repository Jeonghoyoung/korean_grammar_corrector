import time
import uvicorn
from fastapi import FastAPI, Form
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))) # 상위 폴더 내 모듈 참조
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))) # 상위 상위 폴더 내 모듈 참조
from transformer.corrector import Corrector

# pip install fastapi
# pip install uvicorn
# pip install python-multipart # fastapi input 값으로 Text 사용을 위한 모듈

app = FastAPI()

model_path = '../../checkpoint_maxlength70/cp.ckpt'
tokenizer_path = '../../tokenizer/tokenizer_maxlength_70.tok'
max_length = 70

corrector = Corrector(model_path, tokenizer_path, max_length)

@app.post('/repair')
def repair(text: str = Form()):
    s_time = time.time()
    repair_text = corrector.run_repair(text, debug=True)
    print(time.time() - s_time)
    return {'Repair Text': repair_text}

# Restful api 조사, raw model vs keras model 비교(12/15 오전), 작성한 코드 설명.
# 이후 일정 : Multi GPU Training , BERT base 구현

# Command : uvicorn main:app --reload --host=0.0.0.0 --port=8000
# URI : http://127.0.0.1:8000/docs
