"""Fine-tune the victim model on raw or sanitized Part C text."""
import argparse

try:
    from .training import train, training_arguments
except ImportError:
    from training import train, training_arguments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--heldout", required=True)
    parser.add_argument("--target_modules", required=True)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", required=True)
    training_arguments(parser, 4)
    args = parser.parse_args()
    if args.model != "Qwen/Qwen2.5-0.5B":
        parser.error("Victims must use the specified 0.5B base model, not an instruct model.")
    train(args, sanitizer=False)


if __name__ == "__main__":
    main()
