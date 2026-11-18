"""Re-distil the bge-m3 + PCA/SIF student used in the controlled distillation ablation.

Usage (needs `pip install "model2vec[distill]==0.8.1"`; about a minute on a GPU):
    python scripts/distill_bgem3_pca_sif.py --out models/m2v_bgem3_pca256_sif

Plain Model2Vec recipe on the same teacher as potion-multilingual-128M (bge-m3), with
no Tokenlearn step: PCA to 256 dimensions, SIF weighting with coefficient 1e-4, float32
weights. The released table (separate Zenodo upload) was produced with these settings.
"""

from __future__ import annotations

import argparse

from model2vec.distill import distill


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="models/m2v_bgem3_pca256_sif")
    p.add_argument("--device", default=None)
    args = p.parse_args()
    model = distill(
        model_name="BAAI/bge-m3",
        pca_dims=256,
        sif_coefficient=1e-4,
        quantize_to="float32",
        device=args.device,
    )
    model.save_pretrained(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
