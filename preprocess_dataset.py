# Preprocess images in QNRF and NWPU dataset.

import argparse

parser = argparse.ArgumentParser(description='Preprocess')
parser.add_argument('--dataset', default='qnrf',
                    help='dataset name, only support qnrf and nwpu')
parser.add_argument('--input-dataset-path', default='data/QNRF',
                    help='original data directory')
parser.add_argument('--output-dataset-path', default='data/QNRF-Train-Val-Test',
                    help='processed data directory')
args = parser.parse_args()

if args.dataset.lower() == 'qnrf':
    from preprocess.preprocess_dataset_qnrf import main

    main(args.input_dataset_path, args.output_dataset_path, 512, 2048)
elif args.dataset.lower() == 'nwpu':
    from preprocess.preprocess_dataset_nwpu import main

    main(args.input_dataset_path, args.output_dataset_path, 384, 1920)
else:
    raise NotImplementedError
