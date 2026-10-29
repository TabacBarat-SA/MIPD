import os
import httpx
from fastapi import FastAPI,Request,HTTPException
app=FastAPI(title='360 Analytics Pasarela')
TARGETS={'summary':('DATA_URL','http://data:8001'),'customers':('DATA_URL','http://data:8001'),'history':('DATA_URL','http://data:8001'),'segments':('SEGMENTATION_URL','http://segmentation:8003'),'evaluation':('EVALUATION_URL','http://evaluation:8004')}
@app.get('/health')
def health():return {'status':'ok','service':'gateway'}
@app.get('/{path:path}')
async def proxy(path:str,request:Request):
    head=path.split('/')[0]
    if head=='customers' and '/' in path:env,default='MODEL_URL','http://model:8002'
    elif head in TARGETS:env,default=TARGETS[head]
    else:raise HTTPException(404,'Ruta no encontrada')
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response=await client.get(os.environ.get(env,default)+'/'+path,params=request.query_params)
        if response.status_code!=200:raise HTTPException(response.status_code,response.text)
        return response.json()
    except httpx.RequestError as e:raise HTTPException(503,'Servicio no disponible') from e
