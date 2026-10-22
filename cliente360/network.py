"""Red compartida con backprop explicito y Adam, sin dependencias GPU.
La implementacion PyTorch equivalente se encuentra en torch_backend.py.
"""
import numpy as np
from scipy.special import expit

class MultiTask:
    def __init__(self,n_users,n_products,n_features,categories,attrs,dim=16,seed=42):
        r=np.random.default_rng(seed);nc=int(max(categories))+1
        self.categories=np.asarray(categories);self.attrs=np.asarray(attrs);self.dim=dim
        self.p={'U':r.normal(0,.05,(n_users,dim)), 'W':r.normal(0,.12,(n_features,dim)), 'b':np.zeros(dim),
                'V':r.normal(0,.1,(n_products,dim)), 'C':r.normal(0,.1,(nc,dim)), 'A':r.normal(0,.05,(attrs.shape[1],dim)),
                'pb':np.zeros(n_products),'w':r.normal(0,.1,dim),'cb':np.zeros(1)}
        self.p={k:v.astype('float32') for k,v in self.p.items()};self.m={k:np.zeros_like(v) for k,v in self.p.items()};self.v={k:np.zeros_like(v) for k,v in self.p.items()};self.step=0
        self.ymean=0.;self.ystd=1.;self.smear=1.;self.transform="raw"
    def encode(self,x): return np.tanh(self.p['U']+x@self.p['W']+self.p['b'])
    def product_vectors(self): return np.tanh(self.p['V']+self.p['C'][self.categories]+self.attrs@self.p['A'])
    def predict(self,x):
        log=(self.encode(x)@self.p['w']+self.p['cb'][0])*self.ystd+self.ymean
        return np.maximum(0,np.exp(np.clip(log,-10,15))*self.smear-1) if self.transform=="log" else np.maximum(0,log)
    def scores(self,x): return self.encode(x)@self.product_vectors().T/np.sqrt(self.dim)+self.p['pb']
    def update(self,x,y,pairs,labels,lam=1.,clv_weight=1.,lr=.01,decay=1e-4):
        p=self.p;z=self.encode(x);v=self.product_vectors();u,pr=pairs.T
        yp=z@p['w']+p['cb'][0];err=yp-y
        logits=(z[u]*v[pr]).sum(1)/np.sqrt(self.dim)+p['pb'][pr]
        bce=np.mean(np.logaddexp(0,logits)-labels*logits) if len(labels) else 0.
        g={k:np.zeros_like(a) for k,a in p.items()}
        dy=clv_weight*2*err/len(y);dz=dy[:,None]*p['w'];g['w']=z.T@dy;g['cb'][0]=dy.sum()
        ds=lam*(expit(logits)-labels)/max(1,len(labels));dv=np.zeros_like(v)
        np.add.at(dz,u,ds[:,None]*v[pr]/np.sqrt(self.dim))
        np.add.at(dv,pr,ds[:,None]*z[u]/np.sqrt(self.dim));np.add.at(g['pb'],pr,ds)
        dz*=1-z*z;dv*=1-v*v
        g['U']=dz;g['W']=x.T@dz;g['b']=dz.sum(0);g['V']=dv
        np.add.at(g['C'],self.categories,dv);g['A']=self.attrs.T@dv
        self.step+=1
        for k in p:
            grad=g[k]+decay*p[k];grad=np.clip(grad,-5,5)
            self.m[k]=.9*self.m[k]+.1*grad;self.v[k]=.999*self.v[k]+.001*grad*grad
            p[k]-=lr*(self.m[k]/(1-.9**self.step))/(np.sqrt(self.v[k]/(1-.999**self.step))+1e-8)
        return float(np.mean(err**2)),float(bce)
    def save(self,path):
        np.savez_compressed(path,**self.p,categories=self.categories,attrs=self.attrs,meta=np.array([self.ymean,self.ystd,self.smear]),transform=np.array(self.transform))
    @classmethod
    def load(cls,path):
        d=np.load(path);model=cls(len(d['U']),len(d['V']),len(d['W']),d['categories'],d['attrs'],d['U'].shape[1])
        model.p={k:d[k] for k in model.p};model.ymean,model.ystd,model.smear=d['meta'];model.transform=str(d['transform']) if 'transform' in d else 'log';return model

def sample_negatives(positives,history,n_products,n_neg=3,seed=42):
    """No usa interacciones posteriores a la ventana etiquetada."""
    rng=np.random.default_rng(seed);known={}
    for u,p in np.vstack([positives,history]):known.setdefault(int(u),set()).add(int(p))
    negatives=[]
    for u,_ in positives:
        allowed=np.setdiff1d(np.arange(n_products),list(known[int(u)]))
        if len(allowed): negatives.extend((int(u),int(p)) for p in rng.choice(allowed,n_neg))
    neg=np.asarray(negatives,dtype='int64').reshape(-1,2)
    return np.vstack([positives,neg]),np.r_[np.ones(len(positives)),np.zeros(len(neg))].astype('float32')
