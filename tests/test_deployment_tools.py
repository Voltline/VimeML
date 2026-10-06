"""Manual experiment entry points must not start conversion or overwrite evidence."""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ManualToolTests(unittest.TestCase):
    def run_script(self, name, *args):
        return subprocess.run([sys.executable, '-X', 'utf8', str(ROOT / 'scripts/deployment' / name), *map(str, args)],
                              cwd=ROOT, capture_output=True, text=True, encoding='utf-8', timeout=20)

    def test_probe_help_and_import_do_not_load_coreml_or_frozen_weights(self):
        result = self.run_script('ane_output_probe.py', '--help')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('New scratch', result.stdout)
        code = ('import importlib.util,sys; '
                f's=importlib.util.spec_from_file_location("probe", {str(ROOT / "scripts/deployment/ane_output_probe.py")!r}); '
                'm=importlib.util.module_from_spec(s); s.loader.exec_module(m); '
                'assert "coremltools" not in sys.modules; assert "torch" not in sys.modules')
        result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_report_commands_refuse_existing_output_before_reading_input(self):
        (ROOT / 'outputs').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / 'outputs') as temporary:
            folder = Path(temporary)
            output = folder / 'evidence.json'
            output.write_bytes(b'preserved original evidence')
            for script, args in [
                ('summarize_keyboard_trace.py', ['--input', folder / 'absent.xml']),
                ('summarize_memory.py', ['--directory', folder / 'absent']),
            ]:
                with self.subTest(script=script):
                    result = self.run_script(script, *args, '--output', output)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn('Output exists', result.stderr)
                    self.assertEqual(output.read_bytes(), b'preserved original evidence')


if __name__ == '__main__':
    unittest.main()
