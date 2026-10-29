from fastapi import FastAPI,Query
from cliente360 import runtime
app=FastAPI(title='360 Analytics Segmentacion')
@app.get('/health')
def health():return {'status':'ok','service':'segmentation'}
@app.get('/segments')
def segments(source:str=Query('synthetic',pattern='^(synthetic|original)$')):return runtime.segments(source)
