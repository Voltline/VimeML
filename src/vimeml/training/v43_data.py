"""V4.3 mixture support: two web stores with keep masks plus the chat bank."""
import json
from pathlib import Path
import numpy as np

from vimeml.training.data import file_sha, write_json
from vimeml.training.data_v2 import PrefixCropWindowDataset
from vimeml.training.data_mixture import JoinedPrefixWindows
from vimeml.posttraining.streams import ConversionWindows

ROOT = Path(__file__).resolve().parents[3]


def window_sentences(dataset, ids):
    position = np.searchsorted(dataset.end, ids, side='right')
    previous = np.zeros(len(ids), dtype=np.int64)
    present = position > 0
    previous[present] = dataset.end[position[present] - 1] - dataset.sentences[position[present] - 1] - 1
    sentences = ids - previous
    candidates = np.flatnonzero(position < len(dataset.end))
    extra = candidates[ids[candidates] >= dataset.first[position[candidates]]]
    sentences[extra] = dataset.sentences[position[extra]]
    return sentences


def sentence_keys(paths):
    """Canonical text keys of every sentence in the listed TXT files."""
    import hashlib
    import unicodedata
    keys = set()
    for path in paths:
        with Path(path).open(encoding='utf-8') as stream:
            for line in stream:
                text = line.rstrip('\r\n')
                if text:
                    keys.add(hashlib.blake2b(' '.join(unicodedata.normalize('NFKC', text).split()).encode(),
                                            digest_size=16).digest())
    return keys


def prepare_maps_v43(directory, root=ROOT):
    """Rebuild epoch maps for the three-store V4.3 mixture, verifying SHAs."""
    directory, root = Path(directory), Path(root)
    request = json.loads((directory / 'request.json').read_text('utf-8'))
    expected = (json.loads((directory / 'manifest.json').read_text('utf-8'))
                if (directory / 'manifest.json').exists() else None)
    if expected and all((directory / r['file']).exists() and file_sha(directory / r['file']) == r['sha256']
                        for r in expected['epochs']):
        return expected
    bases = []
    for record in request['joined_stores']:
        if (file_sha(root / record['token_dir'] / 'manifest.json') != record['token_manifest_sha256']
                or file_sha(root / record['index_dir'] / 'manifest.json') != record['window_manifest_sha256']):
            raise ValueError('Joined bank identity changed')
        bases.append(PrefixCropWindowDataset(root / record['token_dir'], root / record['index_dir'],
            seed=request['seed'], probability=request['crop_probability'],
            min_remaining_tokens=request['crop_minimum']))
    web_parts = []
    for record, base in zip(request['joined_stores'][:-1], bases[:-1]):
        keep_path = root / record['keep']
        if file_sha(keep_path) != record['keep_sha256']:
            raise ValueError(f"Store keep mask changed: {record['token_dir']}")
        keep = np.load(keep_path, mmap_mode='r', allow_pickle=False)
        chunks = []
        for start in range(0, len(base), 262144):
            ids = np.arange(start, min(len(base), start + 262144), dtype=np.int64)
            chunks.append(ids[keep[window_sentences(base, ids)]])
        web_parts.append(np.concatenate(chunks))
    web = np.concatenate(web_parts)
    joined = JoinedPrefixWindows(bases, request['primary_store'],
                                 [{int(k): v for k, v in r['source_map'].items()} for r in request['joined_stores']])
    chat_base = bases[-1]
    try:
        chat_start = sum(len(base) for base in bases[:-1])
        chat = np.arange(chat_start, chat_start + len(chat_base), dtype=np.int64)
        records = []
        for epoch in range(request['epochs']):
            joined.set_epoch(epoch)
            web_tokens = 0
            for start in range(0, len(web), 262144):
                web_tokens += int(joined.window_lengths(web[start:start + 262144]).sum())
            chat_lengths = joined.window_lengths(chat)
            target = round(web_tokens * request['chat_fraction'] / (1 - request['chat_fraction']))
            rng = np.random.default_rng(request['seed'] + epoch)
            chosen, used = [], 0
            while used < target:
                order = rng.permutation(len(chat))
                cumulative = np.cumsum(chat_lengths[order], dtype=np.int64)
                take = int(np.searchsorted(cumulative, target - used, side='right'))
                if take:
                    chosen.append(chat[order[:take]])
                    used += int(cumulative[take - 1])
                if take < len(order):
                    break
            chosen = np.concatenate(chosen)
            mapping = np.concatenate((web, chosen)).astype('<u4')
            rng.shuffle(mapping)
            path = directory / f'epoch-{epoch}.npy'
            temp = path.with_suffix('.tmp.npy')
            np.save(temp, mapping, allow_pickle=False)
            temp.replace(path)
            record = {'epoch': epoch, 'file': path.name, 'bytes': path.stat().st_size,
                      'sha256': file_sha(path), 'windows': len(mapping),
                      'web_windows': len(web), 'chat_windows': len(chosen),
                      'web_tokens': web_tokens, 'chat_tokens': used,
                      'effective_tokens': web_tokens + used,
                      'chat_fraction': used / (web_tokens + used),
                      'chat_effective_passes': used / int(chat_lengths.sum())}
            if expected and record != expected['epochs'][epoch]:
                raise ValueError(f'Rebuilt map differs from frozen epoch {epoch}')
            records.append(record)
            print(json.dumps({'map_epoch': epoch + 1, 'windows': len(mapping),
                              'chat_fraction': record['chat_fraction']}), flush=True)
        result = {'format': 'vimeml_token_mixture_v1', 'status': 'complete', 'request': request,
                  'epochs': records,
                  'policy': 'V4.3: retained old LM windows and deduplicated expansion windows once per epoch; '
                            'chat-only repeated shuffled cycles at 5% actual cropped web targets; maps SHA256-pinned'}
        if expected and result != expected:
            raise ValueError('Rebuilt mixture manifest differs')
        if expected is None:
            write_json(directory / 'manifest.json', result)
        return result
    finally:
        joined.close()
