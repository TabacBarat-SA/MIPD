"""Aumentacion explicita: originales intactos, nuevas compras con trazabilidad.
No se consultan compras futuras para parametrizar perfiles o catalogo.
"""
import json, hashlib
import numpy as np
import pandas as pd
from .settings import RAW, SYN, REPORT, SEED

def uid(s): return hashlib.md5(s.encode()).hexdigest()
def generate(n_customers=2500, seed=SEED):
    rng = np.random.default_rng(seed)
    names = ['customers', 'orders', 'order_items', 'order_payments', 'products']
    t = {n: pd.read_csv(RAW / f'olist_{n}_dataset.csv') for n in names}
    c,o,it,pay,prod = [t[n] for n in names]
    o['order_purchase_timestamp'] = pd.to_datetime(o.order_purchase_timestamp)
    base = o[(o.order_status=='delivered') & (o.order_purchase_timestamp < '2017-04-01')].merge(c,on='customer_id',validate='many_to_one')
    first = base.sort_values('order_purchase_timestamp').drop_duplicates('customer_unique_id')
    cohort = first.sample(min(n_customers,len(first)),random_state=seed)
    old_items = it[it.order_id.isin(base.order_id)].merge(prod[['product_id','product_category_name']],on='product_id')
    top = old_items.product_category_name.value_counts().head(10).index
    catlog = old_items[old_items.product_category_name.isin(top)].groupby('product_id').agg(category=('product_category_name','first'), price=('price','median'), freight=('freight_value','median'), seller_id=('seller_id','first'), count=('order_id','size')).reset_index()
    catlog = catlog.sort_values('count',ascending=False).groupby('category').head(30).reset_index(drop=True)
    cats = sorted(catlog.category.unique())
    pools = {cat: catlog[catlog.category==cat].index.to_numpy() for cat in cats}
    paid = pay.groupby('order_id').payment_value.sum()
    firstcat = old_items.drop_duplicates('order_id').set_index('order_id').product_category_name
    newc,newo,newit,newpay,manifest,profiles = [],[],[],[],[],[]
    counter=0
    for row in cohort.itertuples():
        person=row.customer_unique_id
        ticket=float(paid.get(row.order_id,120))
        # Propension persistente, correlacionada suavemente con el ticket inicial.
        rate=float(np.clip(rng.lognormal(-.45+.20*np.log(max(ticket,10)/100),.85),.08,3.5))
        budget=float(np.clip(ticket*rng.lognormal(0,.25),25,700))
        preferred=firstcat.get(row.order_id,cats[0])
        if preferred not in cats: preferred=rng.choice(cats)
        dropout=int(rng.integers(5,16)) if rng.random()<.28 else 99
        profiles.append({'customer_unique_id':person,'latent_rate':rate,'latent_budget':budget,'latent_category':preferred,'dropout_month':dropout})
        for month, start in enumerate(pd.date_range('2017-04-01','2018-06-01',freq='MS')):
            if month>=dropout: continue
            season=1.30 if start.month==11 else (1.12 if start.month==12 else 1.)
            count=int(rng.poisson(rate*season))
            for _ in range(count):
                counter+=1; oid=uid(f'synthetic-order-{seed}-{counter}'); cid=uid(f'synthetic-customer-{seed}-{counter}')
                date=start+pd.Timedelta(days=int(rng.integers(0,start.days_in_month)),hours=int(rng.integers(0,24)))
                cat=preferred if rng.random()<.78 else rng.choice(cats)
                pool=pools[cat]
                weights=np.exp(-np.abs(np.log(catlog.loc[pool,'price'].to_numpy()/budget))/.8)*np.sqrt(catlog.loc[pool,'count'].to_numpy())
                pr=catlog.loc[rng.choice(pool,p=weights/weights.sum())]
                price=round(float(pr.price*rng.lognormal(0,.08)),2); freight=round(float(max(3,pr.freight*rng.lognormal(0,.08))),2)
                qty=int(rng.choice([1,2,3],p=[.85,.13,.02])); total=round(qty*(price+freight),2)
                newc.append([cid,person,str(row.customer_zip_code_prefix).zfill(5),row.customer_city,row.customer_state])
                newo.append([oid,cid,'delivered',date,date+pd.Timedelta(hours=2),date+pd.Timedelta(days=2),date+pd.Timedelta(days=7),date+pd.Timedelta(days=12)])
                for q in range(1,qty+1): newit.append([oid,q,pr.product_id,pr.seller_id,date+pd.Timedelta(days=3),price,freight])
                method=rng.choice(['credit_card','boleto','debit_card'],p=[.78,.17,.05])
                installments=int(np.clip(np.ceil(total/100),1,10)) if method=='credit_card' else 1
                newpay.append([oid,1,method,installments,total])
                manifest.append([oid,True,'behavioral_augmentation_v1',seed])
    additions={'customers':pd.DataFrame(newc,columns=c.columns),'orders':pd.DataFrame(newo,columns=o.columns),'order_items':pd.DataFrame(newit,columns=it.columns),'order_payments':pd.DataFrame(newpay,columns=pay.columns)}
    for name,df in additions.items():
        original = pd.read_csv(RAW/f'olist_{name}_dataset.csv', dtype=str)
        pd.concat([original,df],ignore_index=True).to_csv(SYN/f'olist_{name}_dataset.csv',index=False)
    prod.to_csv(SYN/'olist_products_dataset.csv',index=False)
    pd.read_csv(RAW/'product_category_name_translation.csv').to_csv(SYN/'product_category_name_translation.csv',index=False)
    pd.DataFrame(manifest,columns=['order_id','is_synthetic','generator','seed']).to_csv(SYN/'provenance.csv',index=False)
    # Variables latentes SOLO para auditoria, nunca son entradas del modelo.
    pd.DataFrame(profiles).to_csv(SYN/'latent_profiles_AUDIT_ONLY.csv',index=False)
    cohort[['customer_unique_id']].to_csv(SYN/'cohort.csv',index=False)
    catlog[['product_id']].to_csv(SYN/'catalog.csv',index=False)
    original_counts=o.merge(c,on='customer_id').groupby('customer_unique_id').size()
    augmented=pd.concat([o,additions['orders']]).merge(pd.concat([c,additions['customers']]),on='customer_id').groupby('customer_unique_id').size()
    summary={'seed':seed,'cohort_customers':len(cohort),'catalog_products':len(catlog),'original_orders':len(o),'added_orders':len(newo),'original_repeat_fraction':float((original_counts>1).mean()),'augmented_repeat_fraction_all':float((augmented>1).mean()),'cohort_repeat_fraction':float((augmented.reindex(cohort.customer_unique_id)>1).mean()),'warning':'Escenario semisintetico, no evidencia de comportamiento real de Olist.'}
    (REPORT/'synthetic_audit.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    print(json.dumps(summary,indent=2))
    return summary
if __name__=='__main__': generate()
