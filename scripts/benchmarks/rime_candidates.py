"""Real Chinese IME candidates from librime + rime-ice, for the Standard zh benchmark.

The baseline engine is librime (rime.dll, ctypes over rime_get_api) with the rime-ice
schema, patched only where the benchmark needs a fixed, input-only engine:
  - user dictionary off, so no query changes the ranking of a later one;
  - the Lua date/time/lunar/UUID/number/calculator translators and the English
    translators (melt_eng, cn_en) removed, so key sequences such as `rq` or `pain`
    return Chinese words only;
  - emoji and traditionalization off.
Everything else (the rime-ice dictionaries, spelling algebra with abbreviated pinyin,
the Lua filters that reorder candidates) is unchanged.

  deploy  build the dictionaries into <work>/user/build (takes a few minutes)
  probe   print the candidates of a few key sequences
  query   <inputs.jsonl> rows {id, keys, n_han} -> <out.jsonl> rows {id, keys,
          candidates:[...]} where a candidate is kept only if it is all Han and has
          exactly n_han characters, i.e. it converts the whole input (Rime also lists
          candidates for a prefix of the input); the first --n are kept, in Rime order.

The rime-ice checkout is read as data (--shared); the patch files are written into
<work>/user by this script, so the engine configuration is recorded here in Git.
"""
import argparse
import ctypes as C
import json
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[2]
ENGINE=ROOT/'datasets/ime-standard-zh-v1/engine'
HAN=re.compile(r'^[〇㐀-䶿一-鿿豈-﫿\U00020000-\U0003134f]+$')

DEFAULT_CUSTOM='''patch:
  schema_list:
    - schema: rime_ice
  menu/page_size: 20
'''
SCHEMA_CUSTOM='''patch:
  translator/enable_user_dict: false
  engine/translators:
    - punct_translator
    - script_translator
    - table_translator@custom_phrase
'''

API_FIELDS=('setup set_notification_handler initialize finalize start_maintenance is_maintenance_mode '
    'join_maintenance_thread deployer_initialize prebuild deploy deploy_schema deploy_config_file '
    'sync_user_data create_session find_session destroy_session cleanup_stale_sessions '
    'cleanup_all_sessions process_key commit_composition clear_composition get_commit free_commit '
    'get_context free_context get_status free_status set_option get_option set_property get_property '
    'get_schema_list free_schema_list get_current_schema select_schema schema_open config_open '
    'config_close config_get_bool config_get_int config_get_double config_get_string config_get_cstring '
    'config_update_signature config_begin_map config_next config_end simulate_key_sequence '
    'register_module find_module run_task get_shared_data_dir get_user_data_dir get_sync_dir get_user_id '
    'get_user_data_sync_dir config_init config_load_string config_set_bool config_set_int '
    'config_set_double config_set_string config_get_item config_set_item config_clear config_create_list '
    'config_create_map config_list_size config_begin_list get_input get_caret_pos select_candidate '
    'get_version set_caret_pos select_candidate_on_current_page candidate_list_begin candidate_list_next '
    'candidate_list_end user_config_open candidate_list_from_index get_prebuilt_data_dir get_staging_dir '
    'commit_proto context_proto status_proto get_state_label delete_candidate '
    'delete_candidate_on_current_page get_state_label_abbreviated set_input').split()


class Traits(C.Structure):
    _fields_=[('data_size',C.c_int),('shared_data_dir',C.c_char_p),('user_data_dir',C.c_char_p),
              ('distribution_name',C.c_char_p),('distribution_code_name',C.c_char_p),
              ('distribution_version',C.c_char_p),('app_name',C.c_char_p),('modules',C.c_void_p),
              ('min_log_level',C.c_int),('log_dir',C.c_char_p),('prebuilt_data_dir',C.c_char_p),
              ('staging_dir',C.c_char_p)]


class Candidate(C.Structure):
    _fields_=[('text',C.c_char_p),('comment',C.c_char_p),('reserved',C.c_void_p)]


class CandIter(C.Structure):
    _fields_=[('ptr',C.c_void_p),('index',C.c_int),('candidate',Candidate)]


class Api(C.Structure):
    _fields_=[('data_size',C.c_int)]+[(n,C.c_void_p) for n in API_FIELDS]


SID=C.c_size_t
SIGS=dict(setup=(None,C.POINTER(Traits)),initialize=(None,C.POINTER(Traits)),finalize=(None,),
    start_maintenance=(C.c_int,C.c_int),join_maintenance_thread=(None,),create_session=(SID,),
    destroy_session=(C.c_int,SID),select_schema=(C.c_int,SID,C.c_char_p),
    set_option=(None,SID,C.c_char_p,C.c_int),clear_composition=(None,SID),
    set_input=(C.c_int,SID,C.c_char_p),get_input=(C.c_char_p,SID),
    candidate_list_begin=(C.c_int,SID,C.POINTER(CandIter)),
    candidate_list_next=(C.c_int,C.POINTER(CandIter)),candidate_list_end=(None,C.POINTER(CandIter)),
    get_version=(C.c_char_p,))


class Rime:
    def __init__(self,shared,work,dll,log_level=2):
        self.user=Path(work)/'user'; self.user.mkdir(parents=True,exist_ok=True)
        (self.user/'default.custom.yaml').write_text(DEFAULT_CUSTOM,encoding='utf-8')
        (self.user/'rime_ice.custom.yaml').write_text(SCHEMA_CUSTOM,encoding='utf-8')
        empty=Path(work)/'prebuilt-none'; empty.mkdir(exist_ok=True)  # never use a shipped build
        lib=C.CDLL(str(dll)); lib.rime_get_api.restype=C.POINTER(Api)
        raw=lib.rime_get_api().contents
        assert raw.data_size>=C.sizeof(Api)-C.sizeof(C.c_int), 'rime.dll older than rime_api.h'
        self.f={n:C.CFUNCTYPE(s[0],*s[1:])(getattr(raw,n)) for n,s in SIGS.items()}
        self._keep=[str(shared).encode(),str(self.user).encode(),str(empty).encode()]
        t=Traits(); t.data_size=C.sizeof(Traits)-C.sizeof(C.c_int)
        t.shared_data_dir,t.user_data_dir,t.prebuilt_data_dir=self._keep
        t.distribution_name=b'VimeML'; t.distribution_code_name=b'vimeml'; t.distribution_version=b'1'
        t.app_name=b'rime.vimeml'; t.min_log_level=log_level; t.log_dir=b''
        self.traits=t
        self.f['setup'](C.byref(t)); self.f['initialize'](C.byref(t))
        self.version=self.f['get_version']().decode()

    def deploy(self):
        if self.f['start_maintenance'](1): self.f['join_maintenance_thread']()

    def session(self):
        s=self.f['create_session']()
        assert s and self.f['select_schema'](s,b'rime_ice'), 'schema rime_ice not deployed'
        for o in (b'ascii_mode',b'emoji',b'traditionalization',b'full_shape'): self.f['set_option'](s,o,0)
        return s

    def candidates(self,s,keys,limit=300):
        self.f['clear_composition'](s)
        assert self.f['set_input'](s,keys.encode('ascii'))
        out=[]; it=CandIter()
        if self.f['candidate_list_begin'](s,C.byref(it)):
            while len(out)<limit and self.f['candidate_list_next'](C.byref(it)):
                c=it.candidate
                out.append((c.text.decode('utf-8'),(c.comment or b'').decode('utf-8')))
            self.f['candidate_list_end'](C.byref(it))
        self.f['clear_composition'](s)
        return out


def full(cands,n_han,n):
    seen=[];
    for text,_ in cands:
        if len(text)==n_han and HAN.match(text) and text not in seen:
            seen.append(text)
            if len(seen)==n: break
    return seen


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('command',choices=['deploy','probe','query'])
    p.add_argument('args',nargs='*',help='probe: key sequences; query: inputs.jsonl out.jsonl')
    p.add_argument('--shared',default=str(ENGINE/'rime-ice'))
    p.add_argument('--dll',default=str(ENGINE/'librime/dist/lib/rime.dll'))
    p.add_argument('--work',default=str(ROOT/'outputs/benchmarks/ime-standard-zh-v1/rime'))
    p.add_argument('--n',type=int,default=20)
    p.add_argument('--scan',type=int,default=300,help='how many Rime candidates to look through')
    a=p.parse_args()
    r=Rime(a.shared,a.work,a.dll)
    if a.command=='deploy':
        r.deploy(); print(f'librime {r.version}: deployed into {r.user}'); return
    s=r.session()
    if a.command=='probe':
        for k in a.args:
            c=r.candidates(s,k,a.scan)
            print(k,'->',' '.join(t for t,_ in c[:30]))
        return
    src,dst=map(Path,a.args)
    with open(src,encoding='utf-8') as fi, open(dst,'w',encoding='utf-8') as fo:
        for line in fi:
            if not line.strip(): continue
            q=json.loads(line)
            c=full(r.candidates(s,q['keys'],a.scan),q['n_han'],a.n)
            fo.write(json.dumps(dict(id=q['id'],keys=q['keys'],engine=f'librime {r.version} + rime-ice',
                                     candidates=c),ensure_ascii=False)+'\n')


if __name__=='__main__':
    sys.exit(main())
