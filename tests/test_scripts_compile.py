from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ScriptsCompileTest(unittest.TestCase):
    def test_every_script_compiles(self):
        # command-line scripts have no unit tests of their own; a syntax error there only shows on the cluster
        for path in sorted((ROOT / "scripts").rglob("*.py")):
            with self.subTest(path=path.relative_to(ROOT).as_posix()):
                compile(path.read_text("utf-8"), str(path), "exec")


if __name__ == "__main__":
    unittest.main()
