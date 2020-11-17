import argparse
import os
import torch
from train_helper import Trainer
from utils.config import DATASET_LIST,DATASET_PARAMS,ARGS,DATASET_PATHS,DOWNSAMPLE_RATIO
import utils.arg_utils

def parse_args():
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--load-args',
                        help='file to read program args from.'+
                        ' Will ignore other parameters if specified',required=False)
    parser.add_argument('--data-path', default='data/UCF-Train-Val-Test', help='dataset path')
    parser.add_argument('--dataset', help='dataset name', choices=DATASET_LIST,
                           default='qnrf')
    parser.add_argument('--out-path', help='place to save checkpoints and models.', default='./')
    parser.add_argument('--lr', type=float, default='1e-5',
                          help='initial learning rate')
    parser.add_argument('--weight-decay', type=float, default="1e-4",
                          help='the weight decay')
    parser.add_argument('--resume', type=str, default='',
                          help='state dict to resume from. If specified as empty will start over')
    parser.add_argument('--auto-resume', action='store_true',default=False,
                          help="if set will try to find most recent checkpoint in 'out_path' ")
    parser.add_argument('--max-epoch', type=int, default=1000,
                          help='max training epoch')
    parser.add_argument('--val-epoch', type=int, required=False,
                          help='the num of steps to log training information')
    parser.add_argument('--val-start', type=int, default=50,
                          help='the epoch start to val')
    parser.add_argument('--batch-size', type=int, default=10,
                          help='train batch size')
    parser.add_argument('--device', default='0', help='assign device')
    parser.add_argument('--num-workers', type=int, default=3,
                          help='the num of training process')
    parser.add_argument('--wot', type=float, default=0.1, help='weight on OT loss')
    parser.add_argument('--wtv', type=float, default=0.01, help='weight on TV loss')
    parser.add_argument('--reg', type=float, default=10.0,
                          help='entropy regularization in sinkhorn')
    parser.add_argument('--num-of-iter-in-ot', type=int, default=100,
                         help='sinkhorn iterations')
    parser.add_argument('--norm-cood', type=int, default=0, help='Whether to norm cood when computing distance')
    
    
    args = parser.parse_args()

    # load default dataset configurations from datasets/dataset_cfg.json
    def_args_dict = {**DATASET_PARAMS[ARGS['train']['dataset']],
                    **DATASET_PATHS[ARGS['train']['dataset']]}
    # if json file is specified ignore all given options
    if args.load_args:
        def_args_dict.update(**ARGS['train'])
    else: 
        def_args_dict.update(**vars(args))
    # overriding default arguments
    args = argparse.Namespace(**def_args_dict)

    # if auto-resume is specified and not resume is not specified explicitly 
    if not args.resume and 'auto_resume' in args:
        utils.arg_utils.assign_latest_cp(args)

    return args


if __name__ == '__main__':
    args = parse_args()
    torch.backends.cudnn.benchmark = True
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()  # set vis gpu
    trainer = Trainer(args)
    trainer.setup()
    trainer.train()
