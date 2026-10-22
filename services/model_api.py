from fastapi import FastAPI,HTTPException,Query
from cliente360 import runtime
app=FastAPI(title='Cliente360 Modelo')
@app.get('/health')
def health():return {'status':'ok','service':'model'}
@app.get('/customers/{customer_id}')
def customer(customer_id:str,source:str=Query('synthetic',pattern='^(synthetic|original)$'),k:int=Query(10,ge=1,le=50),exclude_seen:bool=False):
    try:return runtime.customer(customer_id,source,k,exclude_seen)
    except KeyError as e:raise HTTPException(status_code=404,detail=str(e)) from e
