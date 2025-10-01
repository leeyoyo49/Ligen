import argparse


def main():
    parser = argparse.ArgumentParser(description="Experiment runner dispatcher")
    parser.add_argument("--config", type=str, default="experiment/spectral.yaml", help="Path to config YAML")
    parser.add_argument(
        "--runner",
        type=str,
        choices=["cgan", "tabgan", "both"],
        default="both",
        help="Which runner to execute"
    )
    args = parser.parse_args()

    if args.runner in ("cgan", "both"):
        from runners.cgan_runner import run as run_cgan
        run_cgan(args.config)
    if args.runner in ("tabgan", "both"):
        from runners.tabgan_runner import run as run_tabgan
        run_tabgan(args.config)


if __name__ == "__main__":
    main()
