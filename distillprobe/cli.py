from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from . import __version__
from .config import RunConfig, build_encoder, load_config
from .modules import REGISTRY
from .reporting import make_run_id, make_run_manifest

log = logging.getLogger("distillprobe")


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def cmd_run(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    if cfg.module not in REGISTRY:
        log.error("unknown module %r; known: %s", cfg.module, sorted(REGISTRY))
        return 2

    log.info("loading encoders: %s", list(cfg.encoders))
    encoders = {name: build_encoder(c) for name, c in cfg.encoders.items()}

    module = REGISTRY[cfg.module]()
    run_id = make_run_id(cfg.model_dump())
    out_dir = Path(args.out or cfg.output_dir) / cfg.module / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = make_run_manifest(cfg.model_dump(), seed=cfg.seed, extra={"run_id": run_id})
    (out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info("run_id=%s out=%s", run_id, out_dir)

    result = module.run(encoders=encoders, data_path=cfg.data_path, config=cfg)
    result.manifest = manifest
    (out_dir / "metrics.json").write_text(
        json.dumps(result.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _dump_csv(result.per_category, out_dir / "per_category.csv")
    _dump_csv(result.tokenization_audit, out_dir / "tokenization_audit.csv")

    plot_paths = module.plot(result.to_dict(), out_dir / "plots")
    if plot_paths:
        log.info("plots: %d written under %s", len(plot_paths), out_dir / "plots")

    log.info("done. results in %s", out_dir)
    return 0


def _dump_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    import pandas as pd  # local import: pandas is in core deps

    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8")


def cmd_list_modules(_: argparse.Namespace) -> int:
    if not REGISTRY:
        print("(no modules registered yet)")
        return 0
    for n in sorted(REGISTRY):
        print(n)
    return 0


def cmd_describe(args: argparse.Namespace) -> int:
    if args.module not in REGISTRY:
        print(f"unknown module: {args.module}", file=sys.stderr)
        return 2
    print(REGISTRY[args.module]().describe())
    return 0


def cmd_sanity_check(args: argparse.Namespace) -> int:
    """Module-aware sanity check."""
    p = Path(args.data_path)
    if not p.exists():
        print(f"not found: {p}", file=sys.stderr)
        return 2
    log.info("inspecting %s", p)
    if p.suffix == ".json":
        try:
            from .data import load_morphology_dataset

            ds = load_morphology_dataset(p)
            print(f"morphology dataset: language={ds.language} version={ds.version}")
            for c in ds.categories:
                print(f"  {c.id:24s} N={len(c.pairs):3d}  {c.label}")
            return 0
        except Exception as e:  # noqa: BLE001
            print(f"could not parse as morphology dataset: {e}", file=sys.stderr)
            return 1
    print(f"(no specific sanity check for {p.suffix} yet)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="distillprobe", description=f"DistillProbe {__version__}")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="run a probing module from a YAML config")
    p_run.add_argument("--config", required=True)
    p_run.add_argument("--out", default=None, help="override output_dir from config")
    p_run.set_defaults(func=cmd_run)

    p_list = sub.add_parser("list-modules", help="list registered modules")
    p_list.set_defaults(func=cmd_list_modules)

    p_desc = sub.add_parser("describe", help="describe a module")
    p_desc.add_argument("module")
    p_desc.set_defaults(func=cmd_describe)

    p_sane = sub.add_parser("sanity-check", help="inspect a data file for known schemas")
    p_sane.add_argument("data_path")
    p_sane.set_defaults(func=cmd_sanity_check)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(getattr(args, "verbose", False))
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
