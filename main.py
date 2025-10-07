import argparse


def main():
    parser = argparse.ArgumentParser(description="Experiment runner dispatcher")
    parser.add_argument("--config", type=str, default="experiment/spectral.yaml", help="Path to config YAML")
    parser.add_argument(
        "--runner",
        type=str,
        choices=["pointgan", "freegan", "both"],
        default="both",
        help="Which runner to execute"
    )
    args = parser.parse_args()

    if args.runner in ("pointgan", "both"):
        from runners.pointgan_runner import run as run_pointgan
        run_pointgan(args.config)
    if args.runner in ("freegan", "both"):
        from runners.freegan_runner import run as run_freegan
        run_freegan(args.config)


if __name__ == "__main__":
    main()
