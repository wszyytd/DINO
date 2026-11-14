"""Pure, denominator-explicit statistics from saved counterfactual records."""
import math
import numpy as np


def describe_scores(scores,identities,seen,label):
    a=np.asarray(scores,dtype=np.float64)
    if a.shape!=(len(identities)+1,) or not np.isfinite(a).all(): raise ValueError('invalid saved scores')
    allowed=np.array([uid in seen for uid in identities]+[True])
    a=np.where(allowed,a,-1e4); p=np.exp(a-a.max()); p/=p.sum()
    ci=int(a.argmax()); choice=identities[ci] if ci<len(identities) else None
    available=[i for i,uid in enumerate(identities) if uid in seen]
    ranking=None; margin=None
    if label in seen and len(available)>=2:
        v=a[available]; cp=np.exp(v-v.max()); cp/=cp.sum()
        target=available.index(identities.index(label)); ranking=available[int(v.argmax())]==identities.index(label)
        margin=float(cp[target]-max(cp[i] for i in range(len(cp)) if i!=target))
    return dict(scores=a.tolist(),probabilities=p.tolist(),choice=choice,label=label,correct=choice==label,
        ranking_correct=ranking,margin=margin,null_probability=float(p[-1]),
        false_reject=choice is None if label is not None else None,
        false_accept=choice is not None if label is None else None)


def rate(n,d):
    return dict(value=n/d if d else None,numerator=n,denominator=d)


def _summary(rows):
    q=[r for r in rows if r['kind']=='query']
    pairs=[r for r in rows if r['kind']=='update' and r['eligible'] and not r['first_write']]
    repeats=[r for r in rows if r['kind']=='repeat']
    help_n=sum(r['full_correct'] and not r['skip_correct'] for r in pairs)
    harm_n=sum(not r['full_correct'] and r['skip_correct'] for r in pairs)
    s=dict(rows=len(rows),eligible_pairs=len(pairs),first_write_pairs=sum(r['kind']=='update' and r['first_write'] for r in rows),
        help_rate=rate(help_n,len(pairs)),harm_rate=rate(harm_n,len(pairs)),net_gain=rate(help_n-harm_n,len(pairs)),
        conditional_harm=rate(harm_n,sum(r['skip_correct'] for r in pairs)),
        query_accuracy=rate(sum(r['correct'] for r in q),len(q)))
    first=[r for r in rows if r['kind']=='update' and r['first_write']]
    s['first_write_full_accuracy']=rate(sum(r['full_correct'] for r in first),len(first))
    history=[r for r in q if r['query_group']==1]
    s['history_query_accuracy']=rate(sum(r['correct'] for r in history),len(history))
    for name,field in (('ranking_accuracy','ranking_correct'),('null_false_reject_rate','false_reject'),('null_false_accept_rate','false_accept')):
        valid=[r[field] for r in q if r.get(field) is not None]; s[name]=rate(sum(valid),len(valid))
    for name in ('margin_delta','null_probability_delta'):
        values=[r[name] for r in pairs if r.get(name) is not None]
        if not all(math.isfinite(x) for x in values): raise ValueError('nonfinite metric input')
        s[name]=rate(sum(values),len(values))
    for name,selected in (('repeat_flip_rate',repeats),('repeat_flip_correct',[r for r in repeats if r['full_correct']]),
                          ('repeat_flip_incorrect',[r for r in repeats if not r['full_correct']])):
        s[name]=rate(sum(r['full_choice']!=r['repeat_choice'] for r in selected),len(selected))
    return s


def summarize_updates(rows):
    strata={}
    fields=('variant','seed','vehicle_uid','view_group','quality','first_write','candidate_complementary','repetitions','scene_id')
    for field in fields:
        groups={}
        for r in rows:
            if field in r: groups.setdefault(str(r[field]),[]).append(r)
        strata[field]={key:_summary(value) for key,value in sorted(groups.items())}
    methods={}
    for r in rows:
        if 'variant' in r: methods.setdefault(r['variant'],{}).setdefault(str(r.get('seed','unknown')),[]).append(r)
    by_method_seed={v:{s:_summary(rs) for s,rs in seeds.items()} for v,seeds in methods.items()}
    by_method_seed_strata={}
    for v,seeds in methods.items():
        by_method_seed_strata[v]={}
        for seed,records in seeds.items():
            sub={}
            for field in fields[2:]:
                groups={}
                for r in records:
                    if field in r: groups.setdefault(str(r[field]),[]).append(r)
                sub[field]={key:_summary(value) for key,value in sorted(groups.items())}
            by_method_seed_strata[v][seed]=sub
    seed_statistics={}
    for v,seeds in by_method_seed.items():
        seed_statistics[v]={}
        for metric in ('query_accuracy','history_query_accuracy','net_gain','harm_rate','repeat_flip_rate'):
            values=[s[metric]['value'] for s in seeds.values() if s[metric]['value'] is not None]
            seed_statistics[v][metric]=dict(mean=float(np.mean(values)) if values else None,
                std=float(np.std(values,ddof=1)) if len(values)>1 else None,seeds=len(values))
    return dict(overall=_summary(rows),strata=strata,by_method_seed=by_method_seed,
        by_method_seed_strata=by_method_seed_strata,seed_statistics=seed_statistics,
        unit='query or (candidate intervention, query) pair; correlated windows are not independent scenes',
        interpretation='Identification help/harm uses correctness, never probability change alone.')
