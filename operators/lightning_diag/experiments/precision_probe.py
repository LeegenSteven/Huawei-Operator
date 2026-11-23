"""Standalone diagnostic for native-input QK and two-component BF16 PV.

Run with the project's CANN environment loaded. This experiment explains why
production PV keeps FP32 precision; it does not change production dispatch.
"""
import argparse
from pathlib import Path
import json
import torch,torch_npu,triton
import triton.language as tl

@triton.jit
def precision_probe(Q,K,V,QK,RECON,O,D:tl.constexpr,FP32_FIRST:tl.constexpr,LOWP_SECOND:tl.constexpr):
 r=tl.arange(0,D)
 q=tl.load(Q+r[:,None]*D+r[None,:]); k=tl.load(K+r[:,None]+r[None,:]*D); v=tl.load(V+r[:,None]*D+r[None,:])
 if FP32_FIRST: q=q.to(tl.float32);k=k.to(tl.float32)
 p=tl.dot(q,k)
 tl.store(QK+r[:,None]*D+r[None,:],p)
 hi=p.to(tl.bfloat16);lo=(p-hi.to(tl.float32)).to(tl.bfloat16)
 tl.store(RECON+r[:,None]*D+r[None,:],hi.to(tl.float32)+lo.to(tl.float32))
 if LOWP_SECOND: out=tl.dot(hi,v)+tl.dot(lo,v)
 else: out=tl.dot(p,v.to(tl.float32))
 tl.store(O+r[:,None]*D+r[None,:],out)

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path)
args=parser.parse_args()
torch.manual_seed(23)
cpu=[torch.randn(64,64).to(torch.bfloat16) for _ in range(3)]
q,k,v=[x.npu() for x in cpu]
ref=cpu[0].double()@cpu[1].double().T
refout=ref@cpu[2].double()
rows=[]
for first in [True,False]:
 for second in [False,True]:
  p=torch.empty((64,64),dtype=torch.float32,device='npu');recon=torch.empty_like(p);out=torch.empty_like(p)
  kernel=precision_probe[(1,)](q,k,v,p,recon,out,64,first,second)
  torch.npu.synchronize(); p=p.cpu().double();recon=recon.cpu().double();out=out.cpu().double()
  row={'fp32_first':first,'lowp_second':second,'qk_abs':(p-ref).abs().max().item(),'qk_rel_l2':((p-ref).norm()/ref.norm()).item(),'recon_abs':(recon-p).abs().max().item(),'out_abs':(out-refout).abs().max().item(),'out_rel_l2':((out-refout).norm()/refout.norm()).item()}
  rows.append(row);print(json.dumps(row),flush=True)
if args.output:
 args.output.parent.mkdir(parents=True,exist_ok=True)
 args.output.write_text(json.dumps(rows,indent=2)+'\n')
