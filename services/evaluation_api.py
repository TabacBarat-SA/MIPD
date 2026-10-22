from fastapi import FastAPI,Query
from cliente360 import runtime
app=FastAPI(title='Cliente360 Evaluacion')
@app.get('/health')
def health():return {'status':'ok','service':'evaluation'}
@app.get('/evaluation')
def evaluation(source:str=Query('synthetic',pattern='^(synthetic|original)$')):return runtime.evaluation(source)
