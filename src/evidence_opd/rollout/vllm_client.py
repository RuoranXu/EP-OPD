"""Local token-only vLLM service; every request is tied to an explicit policy version."""
import atexit
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor


class VLLMRollout:
    def __init__(self,model,folder,physical_gpu='3',utilization=.84,max_sequences=32):
        self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=True,mode=0o700)
        self.socket_path=self.folder/'engine.sock'
        env=os.environ.copy()
        for key in list(env):
            if key in {'RANK','LOCAL_RANK','WORLD_SIZE','LOCAL_WORLD_SIZE','MASTER_ADDR','MASTER_PORT','DEEPSEEK_API_KEY'} or key.startswith('TORCHELASTIC_'):
                env.pop(key,None)
        env.update(CUDA_VISIBLE_DEVICES=physical_gpu,VLLM_USE_V1='0',VLLM_WORKER_MULTIPROC_METHOD='spawn',
                   OMP_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
        script=Path(__file__).resolve().parents[3]/'workers/rollout_service.py'
        with (self.folder/'engine.log').open('ab',buffering=0) as log:
            self.process=subprocess.Popen([sys.executable,'-u',str(script),'--model',str(model),
                '--socket',str(self.socket_path),'--utilization',str(utilization),
                '--max-sequences',str(max_sequences)],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log)
        self.connection=None;self.lock=threading.Lock()
        atexit.register(self.close)
        deadline=time.monotonic()+600
        while time.monotonic()<deadline:
            if self.process.poll() is not None:
                raise RuntimeError('vLLM engine failed; inspect '+str(self.folder/'engine.log'))
            connection=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
            try:connection.connect(str(self.socket_path))
            except (FileNotFoundError,ConnectionRefusedError):
                connection.close();time.sleep(.25);continue
            self.connection=connection;self.stream=connection.makefile('rwb')
            self._rpc({'command':'ping'})
            return
        self.close();raise TimeoutError('vLLM engine startup exceeded 600 seconds')

    def _rpc(self,message):
        with self.lock:
            self.stream.write(json.dumps(message).encode()+b'\n');self.stream.flush()
            line=self.stream.readline()
            if not line:raise RuntimeError('vLLM control socket disconnected')
            result=json.loads(line)
            if not result.get('ok'):raise RuntimeError('vLLM service: '+result.get('error','unknown error'))
            return result

    def reload(self,weights,version):
        return self._rpc({'command':'reload','path':str(Path(weights).resolve()),'policy_version':version})

    def generate_base(self,prompts,seeds,cap,eos,caps):
        result=self._rpc({'command':'generate_base','prompts':prompts,'seeds':seeds,
                          'cap':cap,'caps':caps,'eos':eos,'temperature':0.,'top_p':1.})
        if (result.get('policy_version') is not None or result.get('initial_base') is not True or
                len(result['tokens'])!=len(prompts)):
            raise ValueError('baseline probe was not evaluated on the untouched Base engine')
        return result

    def generate(self,prompts,version,seeds,cap,eos,temperature=.8,top_p=.95,caps=None):
        message={'command':'generate','prompts':prompts,'policy_version':version,
                 'seeds':seeds,'cap':cap,'eos':eos,'temperature':temperature,'top_p':top_p}
        if caps is not None:message['caps']=caps
        result=self._rpc(message)
        if result['policy_version']!=version or len(result['tokens'])!=len(prompts):
            raise ValueError('vLLM policy version or response coverage mismatch')
        return result

    def close(self):
        if getattr(self,'connection',None) is not None:
            try:self.stream.close();self.connection.close()
            except OSError:pass
            self.connection=None
        process=getattr(self,'process',None)
        if process is not None and process.poll() is None:
            process.terminate()
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:process.kill();process.wait()


def packet_batches(client,batches,version,seed,cap,eos,max_documents=32,temperature=.8,top_p=.95,context_window=None):
    """Prefetch one large inference group while the previous group's packets train."""
    groups=[];group=[];documents=0;offset=0
    for index,batch in enumerate(batches):
        count=2*len(batch)
        if group and documents+count>max_documents:
            groups.append(group);group=[];documents=0
        group.append((index,batch,offset));documents+=count;offset+=count
    if group:groups.append(group)
    def generate(group):
        prompts=[];seeds=[];caps=[]
        for index,batch,start in group:
            values=[p['student_prompts'][v] for p in batch for v in ['Dorig','Dcf']]
            prompts.extend(values)
            seeds.extend(seed+version*1000003+start+i for i in range(len(values)))
            if context_window is not None:
                from evidence_opd.rollout.length_budget import batch_cap
                caps.extend([batch_cap(batch,cap,context_window)]*len(values))
        kwargs={'caps':caps} if context_window is not None else {}
        return client.generate(prompts,version,seeds,cap,eos,temperature,top_p,**kwargs)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future=pool.submit(generate,groups[0])
        for i,group in enumerate(groups):
            result=future.result()
            if i+1<len(groups):future=pool.submit(generate,groups[i+1])
            cursor=0
            for index,batch,_ in group:
                count=2*len(batch)
                yield index,result['tokens'][cursor:cursor+count],result
                cursor+=count
