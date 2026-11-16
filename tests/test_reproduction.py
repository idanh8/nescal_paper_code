import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from reproduction.recipes import read_recipe, competitor_policy, bind_data
from reproduction.cli import parser, prepare, main
from protocol_recorder import record_suite


class ProtocolTests(unittest.TestCase):
    def test_original_call_sequences(self):
        reference = json.loads((ROOT / 'tests/reference_traces.json').read_text())
        with tempfile.TemporaryDirectory() as d:
            old = os.getcwd()
            try:
                os.chdir(d)
                for case, expected in reference.items():
                    with self.subTest(case=case):
                        self.assertEqual(record_suite(ROOT / 'src', case), expected)
            finally:
                os.chdir(old)

    def test_explicit_schedule_matches_original_branch_calls(self):
        original = record_suite(ROOT / 'src', 'singleton')['calls']
        for schedule in ('cosine', 'constant'):
            got = record_suite(ROOT / 'src', 'explicit_' + schedule)['calls']
            expected = [c for c in original if c['kwargs']['loss_params']['secondary_loss'] is None
                        or c['kwargs'].get('lambda_kind') == schedule]
            self.assertEqual(got, expected)

    def test_selected_competitor_does_not_run_other_methods(self):
        got = record_suite(ROOT / 'src', 'mmce_only')['calls']
        expected = [c for c in record_suite(ROOT / 'src', 'competitors')['calls']
                    if c['kwargs']['loss_params']['secondary_loss'] == 'mmce']
        self.assertEqual(got, expected)
        self.assertTrue(got)


class LauncherTests(unittest.TestCase):
    def prepare(self, *args, stage='nescal', kind='image'):
        return prepare(parser(stage).parse_args(['--config', str(ROOT / 'configs/examples' / (kind + '.json'))] + list(args)))

    def test_examples_resolve(self):
        for path in (ROOT / 'configs/examples').glob('*.json'):
            prepare(parser().parse_args(['--config', str(path), '--dry-run']))

    def test_explicit_single_schedule_and_seeds(self):
        _, cfg, _, _, _ = self.prepare('--mode', 'explicit', '--onset', '60', '--lambda-cosine', '20', '--seeds', '18', '42')
        self.assertEqual(cfg['candidate_onsets'], [60])
        self.assertEqual(cfg['_explicit_schedules'], ['cosine'])
        self.assertEqual(cfg['lambda_grid_constant'], [])
        self.assertEqual(cfg['stage_b_seeds'], [18, 42])
        self.assertEqual(cfg['confirmatory_seeds'], [])

    def test_synthetic_constant_only(self):
        _, cfg, _, _, _ = self.prepare('--mode', 'explicit', '--onset', '20', '--lambda-constant', '3', kind='synthetic')
        self.assertEqual(cfg['onset_sweep'], [20])
        self.assertEqual(cfg['lambda_grid_constant'], [3])
        self.assertNotIn('lambda_grid_cosine', cfg)

    def test_missing_or_ignored_parameters_rejected(self):
        for args in [('--mode','explicit'), ('--mode','explicit','--onset','60'), ('--onset','60'),
                     ('--mode','explicit','--onset','100','--lambda-cosine','1')]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                self.prepare(*args)

    def test_method_settings_are_required(self):
        with self.assertRaises(ValueError):
            self.prepare('--mode','explicit','--methods','MMCE', stage='competitors')
        recipe, _, _, _, _ = self.prepare('--mode','explicit','--methods','MMCE',
             '--method-params','MMCE={"lambda_reg":7}',stage='competitors')
        specs, fl = competitor_policy(recipe, 'explicit')
        self.assertEqual(list(specs), ['mmce'])
        self.assertEqual(specs['mmce']['search_space'], {'lambda_reg':[7]})
        self.assertEqual(fl, {'fixed_gamma_search':[], 'adaptive_gamma_search':[]})

    def test_saved_results_and_foreign_features_are_not_config_inputs(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'input.json'
            for data in [dict(config={}, reportable={}), dict(config={'embedding_dir':'old/features'}),
                         dict(config={}, selected={'best_onset':60})]:
                p.write_text(json.dumps(data))
                with self.assertRaises(ValueError):read_recipe(p)

    def test_existing_root_still_uses_explicit_config(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'cifar100/resnet50/run';root.mkdir(parents=True)
            (root/'run_manifest.json').write_text('INVALID; MUST NOT BE READ FOR CONFIGURATION')
            _,cfg,stages,_,tag=self.prepare('--run-root',str(root),stage='posthoc')
            self.assertEqual(cfg['base_out'],d)
            self.assertEqual(cfg['dataset'],'cifar100')
            self.assertEqual(stages,['posthoc'])
            self.assertEqual(tag,'run')

    def test_frozen_settings_and_output_path_are_resolved_before_download(self):
        _,cfg,_,root,_=self.prepare('--tag','fresh',kind='frozen')
        self.assertTrue(cfg['early_stopping'])
        self.assertTrue(cfg['finetune'])
        self.assertIn('amazon_multi_5_roberta_finetuned_split42/frozen_mlp/fresh',root)
        self.assertNotIn('embedding_dir',cfg)

    def test_frozen_preparation_builds_features_inside_the_requested_run(self):
        recipe,cfg,_,_,_=self.prepare('--tag','fresh',kind='frozen')
        cfg['model_params']={'arch':'frozen_enc_mlp','num_classes':5,'in_dim':768,'hidden_dims':[128,64]}
        calls=[]
        extractor=types.ModuleType('encoders.extract_embeddings')
        extractor.run=lambda raw,encoder,**kw:(calls.append((raw,encoder,kw)) or kw['output_root']+'/new')
        frozen=types.ModuleType('experiments.frozen');frozen.CONFIG={};frozen._ds_tag=lambda raw,task:raw+'_'+task
        experiments=types.ModuleType('experiments');experiments.__path__=[];experiments.frozen=frozen
        loaders=types.ModuleType('data.frozen_datasets');loaders.load_frozen=lambda *a,**kw:None
        with patch.dict(sys.modules,{'encoders.extract_embeddings':extractor,'experiments':experiments,
                                     'experiments.frozen':frozen,'data.frozen_datasets':loaders}):
            bound=bind_data(recipe,cfg,'runs/example/fresh')
        self.assertEqual(calls[0][2]['output_root'],'runs/example/fresh/features')
        self.assertTrue(calls[0][2]['finetune'])
        self.assertEqual(calls[0][2]['split_seed'],42)
        self.assertEqual(bound['embedding_dir'],'runs/example/fresh/features/new')

    def test_stage_order_and_explicit_configuration(self):
        with tempfile.TemporaryDirectory() as d:
            r=read_recipe(ROOT/'configs/examples/image.json');r['config']['base_out']=d
            path=Path(d)/'config.json';path.write_text(json.dumps(r))
            calls=[]
            package=types.ModuleType('experiments');package.__path__=[]
            ns=types.ModuleType('experiments.nescal');ns.CONFIG={}
            def train(cfg,tag):
                calls.append(('nescal',cfg['dataset'],tag))
                (Path(cfg['base_out'])/cfg['dataset']/cfg['arch_name']/tag/'run_manifest.json').write_text('{}')
            ns.run=train
            co=types.ModuleType('experiments.competitors');co.run_root_ref=['']
            co.run=lambda tag,**kw:calls.append(('competitors',kw['cfg']['dataset'],kw['reuse_baseline_ts']))
            package.nescal=ns;package.competitors=co
            with patch.dict(sys.modules,{'experiments':package,'experiments.nescal':ns,'experiments.competitors':co}), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(['--config',str(path),'--stages','nescal','competitors','--tag','fresh']),0)
            self.assertEqual(calls,[('nescal','cifar100','fresh'),('competitors','cifar100','fresh')])

if __name__ == '__main__':
    unittest.main()
