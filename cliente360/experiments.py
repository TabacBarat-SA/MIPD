"""Sensibilidad OAT: una variable por vez, seleccion SOLO validacion."""
import numpy as np
import pandas as pd
from .settings import OUT,REPORT
from .train import train
from .metrics import regression,ranking

def run(source='synthetic',epochs=100,backend='numpy'):
    d=dict(np.load(OUT/f'{source}.npz'));rows=[]
    configs=[{'lam':v} for v in [.1,.5,1.,2.]]+[{'dim':v} for v in [8,32]]+[{'negatives':v} for v in [1,5]]+[{'decay':v} for v in [0.,.001]]+[{'transform':'log'}]
    for cfg in configs:
        model,curve=train(d,epochs=epochs,backend=backend,**cfg)
        rows.append({'config':str(cfg),'epochs':len(curve),**regression(d['y_val'],model.predict(d['x_val'])),**ranking(model.scores(d['x_val']),d['pairs_val'])})
    pd.DataFrame(rows).to_csv(REPORT/f'{source}_sensitivity.csv',index=False)
