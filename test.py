import argparse
import torch
import os
import numpy as np
import datasets.crowd as crowd
from models import vgg19
from config import

DOWNSAMPLE_RATIO = 8

def parse_args():
    parser = argparse.ArgumentParser(description='Test ')
    load_args = parser.add_mutually_exclusive_group(required = True)
    load_args.add_argument('--load-args',
                        help='file to read program args from.'+
                        ' Cannot be run with other options.'
                        ,required=False)
    default = parser.add_argument_group()
    default.add_argument('--device', default='0', help='assign device')
    default.add_argument('--crop-size', type=int, required=False
                        help='the crop size of the train image')
    default.add_argument('--model-path', type=str, default='pretrained_models/model_qnrf.pth',
                        help='saved model path')
    default.add_argument('--data-path', type=str,
                        default='data/QNRF-Train-Val-Test',
                        help='saved model path')
    default.add_argument('--dataset', type=str, default='qnrf',
                        help='dataset name: qnrf, nwpu, sha, shb')
    default.add_argument('--pred-density-map-path', type=str, default='',
                        help='save predicted density maps when pred-density-map-path is not empty.')

    if args.load_args: # if load args is specified load from the file
        args = argparse.Namespace(**ARGS['test'])

    args = parser.parse_args()
    args = vars(args)
    params_added = {}
    for key,val in DATASET_PARAMS[args["dataset"]].items():
        if not args[key]:
            params_added[key] = val
    args.update(params_added)

    return args


if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()  # set vis gpu
    device = torch.device('cuda')

    model_path = args.model_path
    crop_size = args.crop_size
    data_path = args.data_path


    dataset_name = args.dataset.lower()
    if dataset_name == 'qnrf':
        from datasets.crowd import Crowd_qnrf as Crowd
    elif dataset_name == 'nwpu':
        from datasets.crowd import Crowd_nwpu as Crowd
    elif dataset_name == 'sha':
        from datasets.crowd import Crowd_sh as Crowd
    elif dataset_name == 'shb':
        from datasets.crowd import Crowd_sh as Crowd
    else:
        raise NotImplementedError

    dataset = Crowd(os.path.join(args.data_path,DATASET_PATHS[dataset_name]["val_path"]),
                        crop_size=args.crop_size,
                        downsample_ratio=DOWNSAMPLE_RATIO, method='val')
    dataloader = torch.utils.data.DataLoader(dataset, 1, shuffle=False,
                                            num_workers=1, pin_memory=True)

    if args.pred_density_map_path:
        import cv2
        if not os.path.exists(args.pred_density_map_path):
            os.makedirs(args.pred_density_map_path)

    model = vgg19()
    model.to(device)
    model.load_state_dict(torch.load(model_path, device))
    model.eval()
    image_errs = []
    for inputs, count, name in dataloader:
        inputs = inputs.to(device)
        assert inputs.size(0) == 1, 'the batch size should equal to 1'
        with torch.set_grad_enabled(False):
            outputs, _ = model(inputs)
        img_err = count[0].item() - torch.sum(outputs).item()

        print(name, img_err, count[0].item(), torch.sum(outputs).item())
        image_errs.append(img_err)

        if args.pred_density_map_path:
            vis_img = outputs[0, 0].cpu().numpy()
            # normalize density map values from 0 to 1, then map it to 0-255.
            vis_img = (vis_img - vis_img.min()) / (vis_img.max() - vis_img.min() + 1e-5)
            vis_img = (vis_img * 255).astype(np.uint8)
            vis_img = cv2.applyColorMap(vis_img, cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(args.pred_density_map_path, str(name[0]) + '.png'), vis_img)

    image_errs = np.array(image_errs)
    mse = np.sqrt(np.mean(np.square(image_errs)))
    mae = np.mean(np.abs(image_errs))
    print('{}: mae {}, mse {}\n'.format(model_path, mae, mse))
