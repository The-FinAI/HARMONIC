from transformers.trainer_utils import get_last_checkpoint
import argparse

def main(folder):
    last_checkpoint = get_last_checkpoint(folder)
    return last_checkpoint

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str,
                        default="results/FT-CP/llama2-7b-chat-gen/diabetes", help="folder to save checkpoints")
    args = parser.parse_args()
    main(args.folder)
