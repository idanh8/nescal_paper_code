import ast
from pathlib import Path
import unittest


class SavedParameterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / 'src/handlers/trainer.py'
        tree = ast.parse(path.read_text())
        trainer = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Trainer')
        method = next(n for n in trainer.body if isinstance(n, ast.FunctionDef) and n.name == 'from_saved_params')
        namespace = {}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), namespace)

        def initialize(self, model_params, loss_params, seed=42, optional=None):
            self.settings = dict(model_params=model_params, loss_params=loss_params, seed=seed, optional=optional)

        cls.constructor = type('SavedModel', (), {
            '__init__': initialize, 'from_saved_params': namespace['from_saved_params']})

    def test_saved_values_survive_inactive_legacy_metadata(self):
        settings = {'model_params': {'arch': 'mlp'}, 'loss_params': {'primary_loss': 'XE'},
                    'seed': 613, 'optional': None}
        saved = dict(settings, retired_option=None)
        original = saved.copy()
        model = self.constructor.from_saved_params(saved)
        self.assertEqual(model.settings, settings)
        self.assertIs(model.settings['model_params'], saved['model_params'])
        self.assertEqual(saved, original)
        self.assertEqual(self.constructor.from_saved_params(settings).settings, settings)

    def test_unknown_active_settings_are_not_silently_dropped(self):
        for value in (0, False, 0.4, 'enabled'):
            with self.subTest(value=value), self.assertRaises(TypeError):
                self.constructor.from_saved_params({
                    'model_params': {}, 'loss_params': {}, 'retired_option': value})


if __name__ == '__main__':
    unittest.main()
