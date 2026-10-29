import argparse
from . import synthetic,data,train,evaluate,segmentation

def main():
    p=argparse.ArgumentParser(description='360 Analytics: laboratorio semisintetico')
    p.add_argument('command',choices=['all','generate','prepare','train','evaluate','segment','experiments','extended'])
    p.add_argument('--source',choices=['synthetic','original'],default='synthetic');p.add_argument('--backend',choices=['numpy','torch'],default='numpy');p.add_argument('--epochs',type=int,default=160);p.add_argument('--customers',type=int,default=2500)
    a=p.parse_args()
    if a.command in ['all','generate']:synthetic.generate(a.customers)
    if a.command in ['all','prepare']:data.prepare(a.source)
    if a.command in ['all','train']:train.run(a.source,a.backend,a.epochs)
    if a.command in ['all','evaluate']:evaluate.baselines(a.source)
    if a.command in ['all','segment']:segmentation.segment(a.source)
    if a.command=='extended':
        from .extended import run
        run(a.source)
    if a.command=='experiments':
        from .experiments import run
        run(a.source,a.epochs,a.backend)
if __name__=='__main__':main()
