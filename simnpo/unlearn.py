import sys
import pathlib
BASELINE_PATH = pathlib.Path(__file__).parent.resolve()
sys.path.append(BASELINE_PATH)

from baselines import it_unlearn

import argparse


def main():
    args = get_args()
    print(args.out_dir)
    it_unlearn(
        args.model_dir, args.data_file, args.out_dir,
        retain_data_file=args.retain_data_file,
        loss_type=args.algo,
        per_device_batch_size=args.per_device_batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        max_len=args.max_len,
        tokenizer_dir=args.tokenizer_dir,
        resume_from_checkpoint=args.resume_from_checkpoint,
        beta=args.beta,
        coeff=args.coeff,
        npo_coeff=args.npo_coeff,
        gamma=args.gamma
    )

    return


def get_args():
    parser = argparse.ArgumentParser(description="SimNPO Unlearning")
    parser.add_argument('--algo', type=str)
    parser.add_argument(
        '--model_dir', type=str,
        help="Path to the target model's hf directory."
    )
    parser.add_argument(
        '--tokenizer_dir', type=str, default=None,
        help="Path to the tokenizer's hf directory. Defaults to the target model's directory."
    )
    parser.add_argument(
        '--data_file', type=str,
        help="Path to the forget set file."
    )
    parser.add_argument(
        '--out_dir', type=str,
        help="Path to the output model's hf directory. Creates the directory if it doesn't already exist."
    )
    parser.add_argument(
        '--max_len', type=int, default=4096,
        help="max length of input ids fed to the model"
    )
    parser.add_argument(
        '--resume_from_checkpoint', action='store_true',
    )

    # SimNPO parameters
    parser.add_argument('--per_device_batch_size', type=int, default=2)
    parser.add_argument(
        '--retain_data_file', type=str, default=None,
        help="Path to the retain set file. Required for SimNPO+GDR with non-JSON data."
    )
    parser.add_argument(
        '--lr', type=float, default=1e-5,
        help="Learning rate for SimNPO training."
    )
    parser.add_argument(
        '--epochs', type=int, default=5,
        help="Number of training epochs."
    )

    parser.add_argument(
        '--beta', type=float, default=0.1,
        help="SimNPO beta parameter (temperature)"
    )
    
    parser.add_argument(
        '--coeff', type=float, default=0.1,
        help="Weight for retain loss"
    )

    parser.add_argument(
        '--npo_coeff', type=float, default=0.1,
        help="Weight for forget loss"
    )

    parser.add_argument(
        '--gamma', type=float, default=0.1,
        help="SimNPO gamma parameter"
    )

    args = parser.parse_args()

    if 'gdr' in args.algo and args.data_file and not args.data_file.endswith('.json'):
        assert args.retain_data_file is not None, "SimNPO+GDR selected. Retain set required."

    return args


if __name__ == '__main__':
    main()
