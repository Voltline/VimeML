"""Release inventory and overwrite checks; no Hub access or production inference."""
import hashlib
import json
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/publishing'))
from prepare_hf import prepare, verifier
import hub


class PublishingTests(unittest.TestCase):
    def setUp(self):
        (ROOT / 'outputs').mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / 'outputs')
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        self.check = verifier()

    def fixture(self):
        text = b'license: gpl-2.0\nTest model card\n'
        (self.root / 'README.md').write_bytes(text)
        manifest = {'format': 'vimeml_hf_release_v1', 'license': 'gpl-2.0',
            'repo_id': 'Voltline/vimeml-tiny-ja-v1', 'source_commit': 'a' * 40,
            'files': {'README.md': {'bytes': len(text), 'sha256': hashlib.sha256(text).hexdigest()}}}
        (self.root / 'RELEASE.json').write_text(json.dumps(manifest), encoding='utf-8')
        (self.root / 'SHA256SUMS.txt').write_text(''.join(
            f'{self.check.sha(self.root / name)}  {name}\n' for name in ('README.md', 'RELEASE.json')), encoding='utf-8')

    def test_detects_tampering_and_undeclared_files_but_allows_hub_metadata(self):
        self.fixture()
        _, files = self.check.verify(self.root)
        self.assertEqual(set(files), {'README.md', 'RELEASE.json', 'SHA256SUMS.txt'})
        cache = self.root / '.cache/huggingface'
        cache.mkdir(parents=True)
        (cache / 'download.metadata').write_text('local download cache', encoding='utf-8')
        (self.root / '.gitattributes').write_text('*.pt filter=lfs', encoding='utf-8')
        self.check.verify(self.root)
        (self.root / '.env').write_text('unrelated local file', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Unexpected release files'):
            self.check.verify(self.root)
        (self.root / '.env').unlink()
        (self.root / 'README.md').write_text('altered', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
            self.check.verify(self.root)

    def test_rejects_path_traversal_and_replaced_checksum_inventory(self):
        for name in ('../outside', '/outside', 'a/../../outside', 'a\\b'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.check.safe_path(self.root, name)
        self.fixture()
        (self.root / 'SHA256SUMS.txt').write_text('0' * 64 + '  only-this-file\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'file list differs'):
            self.check.verify(self.root)

    def test_prepare_refuses_existing_directory_without_loading_a_model(self):
        sentinel = self.root / 'evidence.txt'
        sentinel.write_bytes(b'keep existing bytes')
        with self.assertRaisesRegex(ValueError, 'Output exists'):
            prepare(self.root)
        self.assertEqual(sentinel.read_bytes(), b'keep existing bytes')

    def test_upload_uses_exact_inventory_existing_main_and_expected_parent(self):
        self.fixture()
        calls = {}

        class Api:
            def repo_info(self, repo_id, **kwargs):
                calls['info'] = (repo_id, kwargs)
                return SimpleNamespace(sha='b' * 40)

            def create_commit(self, **kwargs):
                calls['commit'] = kwargs
                return SimpleNamespace(oid='c' * 40, commit_url='https://huggingface.co/example/commit/' + 'c' * 40)

        fake = SimpleNamespace(HfApi=Api, CommitOperationAdd=lambda **kwargs: SimpleNamespace(**kwargs))
        with patch.dict(sys.modules, {'huggingface_hub': fake}), patch.object(sys, 'argv',
                ['hub.py', 'upload', '--release', str(self.root)]):
            hub.main()
        commit = calls['commit']
        self.assertEqual(commit['revision'], 'main')
        self.assertEqual(commit['parent_commit'], 'b' * 40)
        self.assertEqual(commit['repo_type'], 'model')
        self.assertEqual({op.path_in_repo for op in commit['operations']}, {'README.md', 'RELEASE.json', 'SHA256SUMS.txt'})


if __name__ == '__main__':
    unittest.main()
