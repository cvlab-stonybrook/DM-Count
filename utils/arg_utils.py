
def assign_latest_cp(args):
    # take a look at the checkpoints at "out_path"
    import os,argparse
    ckpts_path = \
    os.path.join(args.out_path,'ckpts',
                    '{}_input-{}_wot-{}_wtv-{}_reg-{}_nIter-{}_normCood-{}'
                    .format(args.dataset, args.crop_size, args.wot, args.wtv,
                    args.reg, args.num_of_iter_in_ot, args.norm_cood))
    def ckpt_no(f):
        if len(os.path.basename(f).split('.')) >= 2 and \
                os.path.basename(f).split('.')[1] == "tar":
            return int(os.path.basename(f).split('_')[0])
        else:
            return None
    if os.path.exists(ckpts_path):    
        latest_no = max([ckpt_no(f) for f in os.listdir(ckpts_path) if ckpt_no(f) != None])
        if latest_no:
            args = vars(args)
            args.update({'resume':
                        os.path.join(ckpts_path,
                        "{}_ckpt.tar".format(latest_no))})
            args = argparse.Namespace(**args)