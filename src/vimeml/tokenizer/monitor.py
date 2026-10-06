"""TensorBoard events without requiring PyTorch or TensorFlow."""
import json
import re
import time
from pathlib import Path

EM = re.compile(r"EM sub_iter=(\d+) size=(\d+) obj=([\deE+.-]+) num_tokens=(\d+) (?:num_)?tokens/piece=([\deE+.-]+)")
STAGES = {"checking_inputs":0,"training":1,"measuring":2,"encoding":2,"merging":3,"verifying_inputs":3,"complete":4,"failed":-1}


class RunMonitor:
    def __init__(self, output, log_dir, enabled):
        self.output = output
        self.started = time.monotonic()
        self.sequence = self.em_step = 0
        self.writer = None
        if enabled:
            try:
                from tensorboard.summary.writer.event_file_writer import EventFileWriter
            except ImportError:
                raise RuntimeError("请先安装 requirements.txt 中的 TensorBoard 依赖。") from None
            self.writer = EventFileWriter(str(log_dir),flush_secs=5)
        self.log_dir = log_dir

    def scalar(self, tag, value, step):
        if self.writer:
            from tensorboard.compat.proto.event_pb2 import Event
            from tensorboard.compat.proto.summary_pb2 import Summary
            self.writer.add_event(Event(wall_time=time.time(),step=step,
                summary=Summary(value=[Summary.Value(tag=tag,simple_value=float(value))])))

    def update(self, stage, **detail):
        elapsed = time.monotonic()-self.started
        state = {"stage":stage,"elapsed_seconds":elapsed,"tensorboard_dir":str(self.log_dir),**detail}
        temporary = self.output/'progress.tmp'
        temporary.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        temporary.replace(self.output/'progress.json')
        self.scalar('run/stage',STAGES[stage],self.sequence)
        self.scalar('run/elapsed_seconds',elapsed,self.sequence)
        if 'preflight_completed_sentences' in detail:
            self.scalar('inputs/checked_sentences',detail['preflight_completed_sentences'],self.sequence)
            self.scalar('inputs/preflight_percent',100*detail['preflight_completed_sentences']/detail['preflight_total_sentences'],self.sequence)
        if 'completed_sentences' in detail:
            done,total = detail['completed_sentences'],detail['total_sentences']
            prefix = stage if stage in {'encoding','merging'} else ('measurement' if stage=='measuring' else 'run')
            self.scalar(prefix+'/completed_sentences',done,self.sequence)
            self.scalar(prefix+'/percent',100*done/total if total else 100,self.sequence)
            if stage=='encoding':
                self.scalar('encoding/sentences_per_second',detail.get('encoding_sentences_per_second',0),self.sequence)
                if detail.get('encoding_eta_seconds') is not None:
                    self.scalar('encoding/eta_seconds',detail['encoding_eta_seconds'],self.sequence)
            elif stage=='measuring':
                seconds = detail.get('measurement_elapsed_seconds',0)
                self.scalar('measurement/sentences_per_second',done/seconds if seconds else 0,self.sequence)
        if self.writer: self.writer.flush()
        self.sequence += 1

    def native_line(self, line):
        match = EM.search(line)
        if not match: return
        sub_iter,size,obj,tokens,ratio = match.groups()
        for name,value in [('em_objective',obj),('candidate_pieces',size),('tokens_per_piece',ratio),('segmented_tokens',tokens)]:
            self.scalar('tokenizer/'+name,float(value),self.em_step)
        self.update('training',em_updates=self.em_step+1,em_sub_iteration=int(sub_iter),
                    candidate_pieces=int(size),em_objective=float(obj))
        self.em_step += 1

    def final_statistics(self, statistics):
        for split,stats in statistics.items():
            for name in ['characters_per_token','mean_tokens_per_sentence','byte_fallback_token_fraction','roundtrip_mismatches','unknown_tokens']:
                value = stats[name]
                if value is not None: self.scalar(f'corpus/{split}/{name}',value,0)
        if self.writer: self.writer.flush()

    def __enter__(self): return self

    def __exit__(self, kind, error, traceback):
        try:
            if error is not None: self.update('failed',error_type=kind.__name__,error=str(error))
        finally:
            if self.writer: self.writer.close()
