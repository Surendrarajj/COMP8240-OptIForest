"""Generate report tables and figures directly from saved experimental evidence."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

ROOT = Path(__file__).resolve().parents[1]
NAMES = {'ad': 'AD', 'annthyroid': 'Annthyroid', 'breastw': 'BreastW',
         'pima': 'Pima', 'pageblocks': 'PageBlocks', 'yeast': 'Yeast',
         'synthetic': 'Synthetic'}
COLORS = {'OptIForest': '#176b91', 'IsolationForest': '#cf7037'}
MIN_RUNS = {'ad': 5, 'annthyroid': 3, 'pageblocks': 3,
            'breastw': 15, 'pima': 15, 'yeast': 15, 'synthetic': 15}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allow-incomplete', action='store_true')
    args = parser.parse_args()
    result_dir = ROOT / 'results_scaled'
    result_dir.mkdir(exist_ok=True)
    report = ROOT / 'report'
    figures = report / 'figures'
    figures.mkdir(exist_ok=True)
    parts = []
    for name in NAMES:
        path = (ROOT/'results' if name=='ad' else result_dir) / f'{name}_runs.csv'
        if path.exists():
            part = pd.read_csv(path)
            # Keep only paired seeds so every model comparison uses the same
            # random seeds when an interrupted run leaves a partial pair.
            by_model = {m: set(part.loc[part.model.eq(m), 'seed'].astype(int))
                        for m in COLORS}
            paired = sorted(by_model['OptIForest'] & by_model['IsolationForest'])
            part = part[part.seed.astype(int).isin(paired)]
            for model in COLORS:
                seeds = part.loc[part.model.eq(model), 'seed'].tolist()
                if not args.allow_incomplete and (len(seeds)<MIN_RUNS[name] or
                        sorted(seeds)!=list(range(42,42+len(seeds)))):
                    raise ValueError(f'Incomplete final seed sequence for {name}/{model}: {seeds}')
            if not args.allow_incomplete:
                counts=part.groupby('model').seed.count()
                if counts.nunique()!=1:
                    raise ValueError(f'Unpaired final model counts for {name}')
            parts.append(part)
        elif not args.allow_incomplete:
            raise FileNotFoundError(path)
    runs = pd.concat(parts, ignore_index=True)
    runs.to_csv(result_dir / 'all_runs.csv', index=False)
    summary = runs.groupby(['dataset', 'model'], sort=False).agg(
        runs=('seed', 'count'), roc_mean=('roc_auc', 'mean'),
        roc_std=('roc_auc', lambda x: np.std(x, ddof=0)),
        ap_mean=('average_precision', 'mean'),
        ap_std=('average_precision', lambda x: np.std(x, ddof=0)),
        fit_mean=('fit_seconds', 'mean'), score_mean=('score_seconds', 'mean'),
        warnings=('runtime_warning_count', 'sum')).reset_index()
    summary.to_csv(result_dir / 'summary.csv', index=False)
    lookup = summary.set_index(['dataset', 'model'])
    def value(name, model, metric):
        if (name, model) not in lookup.index:
            return r'\textit{pending}'
        r = lookup.loc[(name, model)]
        return f"{r[metric+'_mean']:.4f} $\\pm$ {r[metric+'_std']:.4f}"
    lines = [r'\begin{table*}[t]\centering\small',
             r'\begin{tabular}{lrrrrr}\toprule',
             r'Dataset & Runs & OptIForest ROC-AUC & iForest ROC-AUC & OptIForest AP & iForest AP \\ \midrule']
    for name in NAMES:
        count=int(lookup.loc[(name,'OptIForest'),'runs']) if (name,'OptIForest') in lookup.index else 0
        lines.append(NAMES[name]+' & '+str(count)+' & '+' & '.join([
            value(name, 'OptIForest', 'roc'), value(name, 'IsolationForest', 'roc'),
            value(name, 'OptIForest', 'ap'), value(name, 'IsolationForest', 'ap')])+r' \\')
    lines.extend([r'\bottomrule\end{tabular}',
                  r'\caption{Final seeded evaluation. AD uses its supplied representation. Extension features are standardised. Values are means $\pm$ population standard deviations. Seeds are consecutive from 42, with the exact number of runs shown. AP is average precision. Both models use 100 trees and samples of 512 observations.}\label{tab:results}',
                  r'\end{table*}'])
    (report/'results_table.tex').write_text('\n'.join(lines)+'\n')
    runtime=[r'\begin{table}[t]\centering\small',
             r'\begin{tabular}{lrrr}\toprule',
             r'Dataset & Runs & Fit (s) & Score (s) \\ \midrule']
    for name in NAMES:
        if (name,'OptIForest') not in lookup.index: continue
        r=lookup.loc[(name,'OptIForest')]
        runtime.append(f"{NAMES[name]} & {int(r.runs)} & {r.fit_mean:.2f} & {r.score_mean:.2f} \\\\")
    runtime += [r'\bottomrule\end{tabular}',
                r'\caption{Mean OptIForest wall time per run on the shared evaluation host. Times are environment-specific and are not directly compared with the earlier Mac run.}\label{tab:runtime}',
                r'\end{table}']
    (report/'runtime_table.tex').write_text('\n'.join(runtime)+'\n')
    macros = []
    for name in NAMES:
        if (name, 'OptIForest') in lookup.index:
            r = lookup.loc[(name, 'OptIForest')]
            for key,col in [('Auc','roc_mean'),('Ap','ap_mean')]:
                macros.append('\\newcommand{\\'+name+key+'}{'+f'{r[col]:.4f}'+'}')
        else:
            for key in ['Auc','Ap']:
                macros.append('\\newcommand{\\'+name+key+'}{pending}')
    if ('ad', 'OptIForest') in lookup.index:
        r=lookup.loc[('ad','OptIForest')]
        macros += [r'\newcommand{\adAucDiff}{'+f"{100*(r.roc_mean-.774):+.2f}"+'}',
                   r'\newcommand{\adApDiff}{'+f"{100*(r.ap_mean-.438):+.2f}"+'}']
    (report/'numbers.tex').write_text('\n'.join(macros)+'\n')
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(7,2.65),layout='constrained')
    yy=np.arange(len(NAMES))
    for ax,metric,title in zip(axes,['roc','ap'],['ROC-AUC','Average precision']):
        for model,offset in [('OptIForest',-.12),('IsolationForest',.12)]:
            vals=[];errors=[]
            for name in NAMES:
                if (name,model) in lookup.index:
                    r=lookup.loc[(name,model)];vals.append(r[metric+'_mean']);errors.append(r[metric+'_std'])
                else:vals.append(np.nan);errors.append(0)
            ax.errorbar(vals,yy+offset,xerr=errors,fmt='o',markersize=4,
                        capsize=2,color=COLORS[model],label=model)
        if metric=='roc':ax.axvline(.5,color='.7',ls='--',lw=.8)
        ax.set(yticks=yy,yticklabels=list(NAMES.values()),xlim=(0,1.03),xlabel=title)
        ax.invert_yaxis();ax.grid(axis='x',alpha=.15)
    axes[1].legend(loc='lower right',fontsize=7)
    fig.savefig(figures/'performance.pdf');plt.close(fig)
    detail=pd.read_csv(ROOT/'data_extended/synthetic_detailed.csv')
    fig,ax=plt.subplots(figsize=(3.3,2.4),layout='constrained')
    for kind,color,mark in [('normal','#7197aa','.'),('point','#e39b2e','^'),
                            ('boundary','#309a79','x'),('clustered','#b94f70','s')]:
        d=detail[detail.anomaly_type.eq(kind)]
        ax.scatter(d.feature_1,d.feature_2,s=7 if kind=='normal' else 12,
                   alpha=.6 if kind=='normal' else .85,c=color,marker=mark,label=f'{kind} ({len(d)})',linewidths=.4)
    ax.set(xlabel='Feature 1',ylabel='Feature 2');ax.legend(fontsize=6,ncol=2,loc='lower center')
    fig.savefig(figures/'synthetic.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(7,2.5),layout='constrained')
    for name,col in [('annthyroid','#176b91'),('breastw','#bf7632'),('synthetic','#4b917a')]:
        path=result_dir/f'{name}_OptIForest_seed42.npz'
        if not path.exists():continue
        z=np.load(path);y,s=z['labels'],z['scores']
        fpr,tpr,_=roc_curve(y,s);precision,recall,_=precision_recall_curve(y,s)
        axes[0].plot(fpr,tpr,color=col,label=NAMES[name],lw=1.4)
        axes[1].plot(recall,precision,color=col,label=NAMES[name],lw=1.4)
        axes[1].axhline(y.mean(),ls=':',lw=.7,color=col,alpha=.8)
    axes[0].plot([0,1],[0,1],'--',color='.6',lw=.7)
    axes[0].set(xlabel='False-positive rate',ylabel='True-positive rate',xlim=(0,1),ylim=(0,1.02))
    axes[1].set(xlabel='Recall',ylabel='Precision',xlim=(0,1),ylim=(0,1.02))
    axes[1].legend(fontsize=7,loc='lower left')
    fig.savefig(figures/'curves.pdf');plt.close(fig)
    subtype=[]
    for model in COLORS:
        for seed in range(42,57):
            path=result_dir/f'synthetic_{model}_seed{seed}.npz'
            if not path.exists():continue
            scores=np.load(path)['scores'];top=np.zeros(len(scores),bool)
            top[np.argsort(-scores,kind='stable')[:200]]=True
            for kind in ['normal','point','boundary','clustered']:
                mask=detail.anomaly_type.eq(kind).to_numpy()
                subtype.append(dict(model=model,seed=seed,kind=kind,flagged_fraction=float(top[mask].mean())))
    pd.DataFrame(subtype).to_csv(result_dir/'synthetic_subtypes.csv',index=False)
    sub=pd.DataFrame(subtype).groupby(['model','kind']).flagged_fraction.agg(['mean','std'])
    lines=[r'\begin{table}[t]\centering\small',r'\begin{tabular}{lrr}\toprule',
           r'Group & OptIForest & iForest \\ \midrule']
    for kind in ['point','boundary','clustered','normal']:
        values=[]
        for model in COLORS:
            if (model,kind) in sub.index:
                r=sub.loc[(model,kind)];values.append(f'{100*r["mean"]:.1f}\\%')
            else:values.append('pending')
        lines.append(kind.title()+' & '+' & '.join(values)+r' \\')
    lines += [r'\bottomrule\end{tabular}',r'\caption{Mean fraction flagged in each synthetic group when the 200 highest-scoring records are selected. For anomaly groups this is recall. For normal records it is the false-positive rate.}\label{tab:subtypes}',r'\end{table}']
    (report/'subtype_table.tex').write_text('\n'.join(lines)+'\n')
    print(summary.to_string(index=False))


if __name__=='__main__':
    main()
