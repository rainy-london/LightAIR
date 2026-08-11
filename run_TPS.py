import os
import argparse


# Set it correctly for distributed training across nodes
NNODES = 1
NODE_RANK = 0
MASTER_ADDR = '127.0.0.1'
MASTER_PORT = 1624  # 0~65536
NPROC_PER_NODE = 4  # e.g. 4 gpus

print("NNODES, ", NNODES)
print("NODE_RANK, ", NODE_RANK)
print("MASTER_ADDR, ", MASTER_ADDR)
print("MASTER_PORT, ", MASTER_PORT)
print("NPROC_PER_NODE, ", NPROC_PER_NODE)


def get_dist_launch(args):
    if args.dist == 'f4':
        return "CUDA_VISIBLE_DEVICES=0,1,2,3 WORLD_SIZE=4 python3 -m torch.distributed.run --nproc_per_node=4 " \
               "--nnodes={:} --node_rank={:} " \
               "--master_addr={:} --master_port={:} ".format(NNODES, NODE_RANK, MASTER_ADDR, MASTER_PORT)
    elif args.dist == 'f2':
        return "CUDA_VISIBLE_DEVICES=0,1 WORLD_SIZE=2 python3 -m torch.distributed.run --nproc_per_node=2 " \
               "--nnodes={:} --node_rank={:} " \
               "--master_addr={:} --master_port={:} ".format(NNODES, NODE_RANK, MASTER_ADDR, MASTER_PORT)
    elif args.dist.startswith('gpu'):  # use one gpu, --dist "gpu0"
        num = int(args.dist[3:])
        return "CUDA_VISIBLE_DEVICES={:} WORLD_SIZE=1 python3 -m torch.distributed.run --nproc_per_node=1 " \
               "--nnodes={:} --node_rank={:} " \
               "--master_addr={:} --master_port={:} ".format(num, NNODES, NODE_RANK, MASTER_ADDR, MASTER_PORT)
    else:
        raise ValueError


def run(args):
    args.config = 'configs/' + args.task + '_X2VLM.yaml'
    print(args.task, args.config)

    dist_launch = get_dist_launch(args)
    os.system(
        f"{dist_launch} Search_TPS_X2VLM.py --config {args.config} --task {args.task} --output_dir {args.output_dir} "
        f"--checkpoint {args.checkpoint} --bs {args.bs} --epo {args.epo} --lr {args.lr} --seed {args.seed} "
        f"--action_temp {args.action_temp} --itm_weight {args.itm_weight} --cls_weight {args.cls_weight} "
        f"--top_k {args.top_k} --entropy_tau {args.entropy_tau} "
        f"{'--evaluate' if args.evaluate else ''}" )


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='tpas', type=str)
    parser.add_argument('--dist', default='gpu0', type=str, help="see func get_dist_launch for details")
    parser.add_argument('--output_dir', default='out/cmp', type=str, help='local path')
    parser.add_argument('--checkpoint', default='checkpoint/x2vlm_base_1b.th', type=str)
    parser.add_argument('--bs', default=0, type=int, help="mini batch size")
    parser.add_argument('--epo', default=0, type=int, help="epoch")
    parser.add_argument('--lr', default=0.0, type=float)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--evaluate', action='store_true', help="directly evaluation")
    parser.add_argument('--action_temp', default=0.1, type=float)
    parser.add_argument('--itm_weight', default=4.0, type=float)
    parser.add_argument('--cls_weight', default=1.0, type=float)
    parser.add_argument('--top_k', default=3, type=int)
    parser.add_argument('--entropy_tau', default=1.0, type=float)
    args = parser.parse_args()

    run(args)
