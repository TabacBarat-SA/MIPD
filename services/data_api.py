from fastapi import FastAPI,Query
from cliente360 import runtime
app=FastAPI(title='Cliente360 Datos')
@app.get('/health')
def health():return {'status':'ok','service':'data'}
@app.get('/summary')
def summary(source:str=Query('synthetic',pattern='^(synthetic|original)$')):return runtime.summary(source)
@app.get('/customers')
def customers(source:str=Query('synthetic',pattern='^(synthetic|original)$')):return runtime.customers(source)
@app.get('/history/{customer_id}')
def history(customer_id:str,source:str=Query('synthetic',pattern='^(synthetic|original)$')):return runtime.history(customer_id,source)
