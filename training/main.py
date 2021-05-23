from solver import Solver
import os
import argparse


def main(config):
    solver = Solver(config)
    solver.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--conv_channels", type=int, default=128)
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--n_fft", type=int, default=512)
    parser.add_argument("--n_harmonic", type=int, default=6)
    parser.add_argument("--semitone_scale", type=int, default=2)
    parser.add_argument(
        "--learn_bw", type=str, default="only_Q", choices=["only_Q", "fix"]
    )

    parser.add_argument("--input_length", type=int, default=80000)

    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--log_step", type=int, default=19)

    parser.add_argument("--model_save_path", type=str, default="./../saved_models")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument(
        "--data_path", type=str, default="./../../jetatag/dataset"
    )
    config = parser.parse_args()

    print(config)
    main(config)
