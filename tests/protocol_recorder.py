import ast
from copy import deepcopy
import contextlib
import io
import os
from pathlib import Path
import types
import numpy as np


def load_functions(path, env):
    tree=ast.parse(Path(path).read_text())
    tree.body=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))]
    exec(compile(tree,str(path),'exec'),env)
    return types.SimpleNamespace(**env)


def normalized(x):
    if callable(x):return '<loader-factory>'
    if isinstance(x,dict):return {str(k):normalized(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [normalized(v) for v in x]
    if isinstance(x,(np.floating,np.integer)):return x.item()
    return x


def record_suite(src, case):
    src=Path(src)
    legacy=(src/'runners').exists()
    paths={'nescal':'run_nescal','competitors':'run_competitors','synthetic':'run_synthetic',
           'ablations':'run_ablations','perturbations_matched':'run_wave5_full'}
    def path(name):
        return src/('runners' if legacy else 'experiments')/(paths[name]+'.py' if legacy else name+'.py')
    calls=[];manifests={}
    class Tracker:
        def __init__(self,*a,**kw):pass
    def run_dir(base,ds,arch,ts,method,seed):return f'{base}/{ds}/{arch}/{ts}/{method}/seed_{seed}'
    def train_one(tracker,*,label,epochs_to_train,run_kwargs):
        calls.append({'op':'train','epochs':epochs_to_train,'kwargs':normalized(run_kwargs)})
        loss=run_kwargs['loss_params'];lam=loss.get('lambda_reg',0)
        onset=(run_kwargs.get('activation_params') or {}).get('fixed_epoch',0)
        score=(lam-2)**2*.01+(onset-3)**2*.001+run_kwargs['seed']*.00001
        return {'nll':score},{'nll':score},False,run_kwargs['root_path']
    def select(scores):
        t={k:sum(v.values())/len(v) if v and all(x is not None for x in v.values()) else float('inf') for k,v in scores.items()}
        return min(t,key=t.get),t
    def write_manifest(p,obj):manifests[p]=normalized(deepcopy(obj))
    def loaders(*a,**kw):
        seed=kw.get('seed',a[1] if len(a)>1 else 0)
        return tuple(f'{x}:{seed}' for x in ('train','val','test','embedding','names'))
    cfg={'dataset':'cifar100','arch_name':'resnet50','model_params':{'arch':'resnet50','num_classes':100},
         'num_epochs':8,'batch_size':4,'lr':.1,'momentum':.9,'weight_decay':.0005,'num_bins':15,
         'diag_every_n':10,'num_workers':0,'data_root':'data','base_out':'runs','candidate_onsets':[2,3],
         'stage_b_seeds':[18,42],'confirmatory_seeds':[613], 'baseline_checkpoint_every':2,
         'selection_mode':case if case in ('joint','sequential') else 'joint', 'lambda_ref':1,
         'lambda_grid_cosine':[1,2],'lambda_grid_constant':[1,2],
         'nescal':{'k':5,'mask_mode':'none','divergence_type':'js','weight_type':'topk','tau':1},
         'ablation_seeds':[18,42], 'ablations':{'divergence':['js','fisher'],'mask':['hard'],'k':[2],
                                             'graph':['pca16'],'penalty_target':3.0}}
    env={'os':os,'np':np,'copy':__import__('copy'),'itertools':__import__('itertools'),
         'CONFIG':cfg,'RunTracker':Tracker,'train_one':train_one,'run_dir':run_dir,'timestamp':lambda:'test',
         'select_by_val_nll':select,'write_manifest':write_manifest,'load_dataset':loaders,
         'send_notification':lambda *a:None,'load_json':lambda p:deepcopy(manifests[p]),
         'build_perturbed_cifar_loaders':loaders,'load_synthetic':loaders}
    rn=load_functions(path('nescal'),dict(env))
    env['run_nescal']=rn
    spec_env={};exec((src/'configs/method_specs.py').read_text(),spec_env)
    specs=spec_env['method_specs'];fl=spec_env['fl_primary_spec']
    for sp in specs.values():
        for key in ('search_space','fallback_search_space'):
            if key in sp:sp[key]={k:v[:2] for k,v in sp[key].items()}
    ce=dict(env,method_specs=specs,fl_primary_spec=fl,run_root_ref=['runs/cifar100/resnet50/test'])
    comp=load_functions(path('competitors'),ce)
    env['run_competitors']=comp
    env['run_posthoc']=types.SimpleNamespace(POSTHOC=(),run=lambda *a,**kw:calls.append({'op':'posthoc','args':normalized(a),'kwargs':normalized(kw)}))
    root='runs/cifar100/resnet50/test'
    with contextlib.redirect_stdout(io.StringIO()):
        if case == 'singleton' or case.startswith('explicit_'):
            cfg['candidate_onsets']=[3]
            cfg['lambda_grid_cosine']=[2]
            cfg['lambda_grid_constant']=[2]
            if case == 'singleton':
                rn.run(cfg,'test')
            else:
                schedule=case.removeprefix('explicit_')
                cfg['lambda_grid_' + ('constant' if schedule=='cosine' else 'cosine')]=[]
                rn.run(cfg,'test',schedules=[schedule])
        elif case == 'mmce_only':
            comp.run('test',cfg,reuse_baseline_ts='test',specs={'mmce':specs['mmce']},
                     fl_spec={'fixed_gamma_search':[],'adaptive_gamma_search':[]})
        elif case in ('joint','sequential'):
            rn.run(cfg,'test')
        elif case=='competitors':
            comp.run('test',cfg,reuse_baseline_ts='test')
        elif case=='competitors_fixed':
            cfg['dataset']='cifar10'
            comp.run('test',cfg,reuse_baseline_ts='test')
        elif case in ('ablations','perturbations_matched'):
            manifests[root+'/run_manifest.json']={'best_onset':3,'lambda_cosine':2}
            if case=='ablations':
                load_functions(path(case),dict(env)).run(root,cfg)
            else:
                cfg['dataset']='cifar10';rn.CONFIG.update(cfg)
                opts={'DATASET':'cifar10','ARCH':'resnet50','RUN_ROOT':root,'LT_LAMBDA':1.0,
                      'LT_ARMS':{'js':'js','fisher':'fisher','uniform':'l2'},'LT_FACTORS':[10],
                      'NOISE_ARMS':{'js':'js'},'NOISE_RATES':[.2],'SEEDS':[18],
                      'EPS_FISHER':.01,'SKIP_DONE':False,'MPATH_NAME':'wave5_fisherfix_manifest.json'}
                load_functions(path(case),dict(env,**opts)).run()
        elif case in ('synthetic','synthetic_only'):
            cfg.update(dataset='synthetic',arch_name='mlp',schedule='constant',onset_sweep=[2,3],
                       optimizer='adam',use_amp=False,select_metric='val_nll',blob={'heterogeneity_strength':.4},
                       model_params={'arch':'mlp','num_classes':3},ablations={'divergence':['js','fisher'],'penalty_target':.05})
            syn=load_functions(path('synthetic'),dict(env))
            if case=='synthetic':syn.run_headline(cfg,'test')
            else:syn.run_headline(cfg,'test',include_followups=False)
    return {'calls':calls,'manifests':manifests}
