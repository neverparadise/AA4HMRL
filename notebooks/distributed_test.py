import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def init_for_distributed(rank, args):
    # ? 1. 분산 훈련을 위한 세팅
    """
    node: GPU가 장착된 machine
    rank: proecess ID. 메인함수에서 rank를 자동으로 배정?
    - local rank: 노드 내부에서 process의 ID [0, L-1]
    - glocal rank: 전체 world 에서 process의 ID [0, W-1]
    world size: 분산 환경에서 실행되는 총 프로세스의 수 또는 
        사용할 총 gpu의 개수: node x num_gpus per node
    - local world size: 각 노드에서 실행되는 총 프로세스의 수 (gpu 개수)
    num_workers: 
    
    하나의 프로세스가 0.25개 또는 0.5개의 gpu를 점유하는 상황도 가능할가? -> 불가능하다
    커널 연산은 하나의 프로세스에서만 사용 가능하다.
    
    멀티프로세싱 큐에 보내진 텐서들은 공유된 메모리로 이동함.
    텐서가 다른 프로세스에 보내지면 텐서값과 그래디언트 모두 공유됨.
    """
    args.rank = rank
    args.num_workers = 4 # len(args.gpu_ids) * 4
    args.world_size =len(args.gpu_ids)
    local_gpu_id = int(args.gpu_ids[args.rank])
    torch.cuda.set_device(local_gpu_id)
    if args.rank is not None:
        print(f"Use GPU: {local_gpu_id} for training")
    dist.init_process_group(backend='nccl',
                            init_method='tcp://127.0.0.1:23456',
                            world_size=args.world_size,
                            rank=args.rank,
                            )
    # ? 아래함수를 실행하면 모든 프로세스가 동기화된다. 
    torch.distributed.barrier()  
    is_master = args.rank == 0
    setup_for_distributed(is_master)  
    print(args)
    
    model = model.cuda(args.rank)
    model = DDP(module=model,
            device_ids=[args.rank])
    
    
def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print
    
