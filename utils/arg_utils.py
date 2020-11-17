def assign_latest_cp(args):
    """
    From the given arguments checks the 'out_path' parameters.
    Looks for a checkpoint assuming the checkpoint was saved in out_path
    and the previous run was with the same training parameters.
    
    Example:
        If you run the program with these arguments;
        "dataset = sha,
        crop_size = 256,
        wot = 0.1,
        wtv = 0.01,
        reg = 10.0,
        num_of_iter_in_ot = 100,
        norm_cood = 0"
        out_path = /DM-Count
        
        After training some epochs when auto_resume is specified 
        this function will look at this directort 
        /DM-Count/ckpts/sha_input-256_wot-0.1_wtv-0.01_reg-10.0_nIter-100_normCood-0/
        
        Assuming checkpoints are saved in "x_ckpt.tar" format this function would assign
        "/DM-Count/ckpts/sha_input-256_wot-0.1_wtv-0.01_reg-10.0_nIter-100_normCood-0/7_ckpt.tar"
        path to args.resume
    """
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
        ckpt_list = [ckpt_no(f) for f in os.listdir(ckpts_path) if ckpt_no(f) != None]
        if ckpt_list:
            latest_no = max(ckpt_list)
            args = vars(args)
            args.update({'resume':
                        os.path.join(ckpts_path,
                        "{}_ckpt.tar".format(latest_no))})
            args = argparse.Namespace(**args)