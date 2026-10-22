import numpy as np
from scipy.stats import spearmanr

def regression(y,p):
    n=max(1,int(np.ceil(len(y)*.1)));top=np.argsort(-p,kind='stable')[:n]
    rho=float(spearmanr(y,p).statistic) if np.std(y)>0 and np.std(p)>0 else 0.
    return {'MAE':float(np.mean(abs(y-p))),'RMSE':float(np.sqrt(np.mean((y-p)**2))),'Spearman':rho,'lift_top10':float(y[top].mean()/y.mean()) if y.mean()>0 else 0.}

def ranking(scores,pairs,k=10,mask=None):
    scores=np.array(scores,copy=True)
    if mask is not None and len(mask):scores[mask[:,0],mask[:,1]]=-np.inf
    relevant={}
    for u,p in pairs:relevant.setdefault(int(u),set()).add(int(p))
    precision=[];recall=[];ndcg=[];catalog=set();k=min(k,scores.shape[1])
    for u,truth in relevant.items():
        rank=np.argsort(-scores[u],kind='stable')[:k];hits=np.array([p in truth for p in rank]);catalog.update(rank.tolist())
        precision.append(hits.sum()/k);recall.append(hits.sum()/len(truth))
        discounts=1/np.log2(np.arange(k)+2);ideal=discounts[:min(len(truth),k)].sum();ndcg.append(float((hits*discounts).sum()/ideal))
    return {f'Precision@{k}':float(np.mean(precision)) if precision else 0.,f'Recall@{k}':float(np.mean(recall)) if recall else 0.,f'NDCG@{k}':float(np.mean(ndcg)) if ndcg else 0.,'coverage':len(catalog)/scores.shape[1],'eligible_users':len(relevant)}
