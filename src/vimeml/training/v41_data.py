"""Frozen V4.1 virtual union; deterministic maps can be rebuilt on AutoDL."""
import json
from pathlib import Path
import numpy as np

from vimeml.training.data import file_sha, write_json
from vimeml.training.data_v2 import PrefixCropWindowDataset
from vimeml.training.data_mixture import JoinedPrefixWindows
from vimeml.posttraining.streams import ConversionWindows

ROOT=Path(__file__).resolve().parents[3]


def window_sentences(dataset, ids):
    position=np.searchsorted(dataset.end,ids,side='right')
    previous=np.zeros(len(ids),dtype=np.int64)
    present=position>0
    previous[present]=dataset.end[position[present]-1]-dataset.sentences[position[present]-1]-1
    sentences=ids-previous
    candidates=np.flatnonzero(position<len(dataset.end))
    extra=candidates[ids[candidates]>=dataset.first[position[candidates]]]
    sentences[extra]=dataset.sentences[position[extra]]
    return sentences


def prepare_maps(directory, root=ROOT):
    directory,root=Path(directory),Path(root)
    request=json.loads((directory/'request.json').read_text('utf-8'))
    expected=json.loads((directory/'manifest.json').read_text('utf-8')) if (directory/'manifest.json').exists() else None
    if expected and all((directory/r['file']).exists() and file_sha(directory/r['file'])==r['sha256'] for r in expected['epochs']):
        return expected
    keep_path=root/request['old_keep']
    if file_sha(keep_path)!=request['old_keep_sha256']:
        raise ValueError('Old training exclusion mask changed')
    keep=np.load(keep_path,mmap_mode='r',allow_pickle=False)
    bases=[]
    for record in request['joined_stores']:
        if file_sha(root/record['token_dir']/'manifest.json')!=record['token_manifest_sha256'] or file_sha(root/record['index_dir']/'manifest.json')!=record['window_manifest_sha256']:
            raise ValueError('Joined bank identity changed')
        bases.append(PrefixCropWindowDataset(root/record['token_dir'],root/record['index_dir'],
            seed=request['seed'],probability=request['crop_probability'],min_remaining_tokens=request['crop_minimum']))
    joined=JoinedPrefixWindows(bases,0,[{int(k):v for k,v in r['source_map'].items()} for r in request['joined_stores']])
    try:
        chunks=[]
        for start in range(0,len(bases[0]),262144):
            ids=np.arange(start,min(len(bases[0]),start+262144),dtype=np.int64)
            chunks.append(ids[keep[window_sentences(bases[0],ids)]])
        web=np.concatenate(chunks)
        chat=np.arange(len(bases[0]),len(joined),dtype=np.int64)
        records=[]
        for epoch in range(request['epochs']):
            joined.set_epoch(epoch)
            web_tokens=0
            for start in range(0,len(web),262144):
                web_tokens+=int(joined.window_lengths(web[start:start+262144]).sum())
            chat_lengths=joined.window_lengths(chat)
            target=round(web_tokens*request['chat_fraction']/(1-request['chat_fraction']))
            rng=np.random.default_rng(request['seed']+epoch)
            chosen,used=[],0
            while used<target:
                order=rng.permutation(len(chat))
                cumulative=np.cumsum(chat_lengths[order],dtype=np.int64)
                take=int(np.searchsorted(cumulative,target-used,side='right'))
                if take:
                    chosen.append(chat[order[:take]])
                    used+=int(cumulative[take-1])
                if take<len(order):
                    break
            chosen=np.concatenate(chosen)
            mapping=np.concatenate((web,chosen)).astype('<u4')
            rng.shuffle(mapping)
            path=directory/f'epoch-{epoch}.npy'
            temp=path.with_suffix('.tmp.npy')
            np.save(temp,mapping,allow_pickle=False)
            temp.replace(path)
            record={'epoch':epoch,'file':path.name,'bytes':path.stat().st_size,'sha256':file_sha(path),
                'windows':len(mapping),'web_windows':len(web),'chat_windows':len(chosen),
                'web_tokens':web_tokens,'chat_tokens':used,'effective_tokens':web_tokens+used,
                'chat_fraction':used/(web_tokens+used),'chat_effective_passes':used/int(chat_lengths.sum())}
            if expected and record!=expected['epochs'][epoch]:
                raise ValueError(f'Rebuilt map differs from frozen epoch {epoch}')
            records.append(record)
            print(json.dumps({'map_epoch':epoch+1,'windows':len(mapping),'chat_fraction':record['chat_fraction']}),flush=True)
        result={'format':'vimeml_token_mixture_v1','status':'complete','request':request,'epochs':records,
                'policy':'Each retained V2.1/V4 LM window once per epoch; chat-only repeated shuffled cycles at 5% actual cropped LM targets; maps SHA256-pinned'}
        if expected and result!=expected:
            raise ValueError('Rebuilt mixture manifest differs')
        # Preserve the frozen bytes across Windows/Linux newline conventions.
        # Rebuilding maps must not rewrite an existing manifest with a new SHA.
        if expected is None:
            write_json(directory/'manifest.json',result)
        return result
    finally:
        joined.close()


def conversion_quota(cursor, updates, examples):
    if not 0<=cursor<updates or updates<1 or examples<updates:
        raise ValueError('Conversion epoch cannot be spread across these updates')
    return ((cursor+1)*examples)//updates-(cursor*examples)//updates


class IndexedConversionWindows(ConversionWindows):
    def __init__(self,data,index_path):
        super().__init__(data)
        self.index_path=Path(index_path)
        self.indices=np.load(self.index_path,mmap_mode='r',allow_pickle=False)
        if self.indices.dtype!=np.dtype('<u4') or self.indices.ndim!=1 or not len(self.indices):
            raise ValueError('Invalid frozen conversion subset')

    def __len__(self):
        return len(self.indices)

    def __getitem__(self,index):
        return self.data[int(self.indices[int(index)])]

    def window_lengths(self,indices):
        return super().window_lengths(self.indices[np.asarray(indices,dtype=np.int64)])

    def __getstate__(self):
        return {'data':self.data,'index_path':self.index_path}

    def __setstate__(self,state):
        self.__init__(**state)
