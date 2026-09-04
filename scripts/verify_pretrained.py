"""Verify pretrained weight parity without pytest.

Loads reference weights (timm ViT / torchvision ResNet-18), remaps them into
the test-only compat assemblies (imported from ``tests/test_weight_parity.py``
— full nets stay out of ``src`` by design), and reports max abs diff /
cosine similarity / prediction agreement.

Usage::

    uv run python scripts/verify_pretrained.py [--arch vit_base|resnet18|all]
                                               [--batch-size 8] [--num-batches 4]
                                               [--write-results]

``--write-results`` refreshes the parity table in ``RESULTS.md``.
Network access is required (downloads reference weights once, then cached).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from test_weight_parity import (  # noqa: E402
    CompatResNet18,
    CompatViT,
    parity_metrics,
)

# Published top-1 (ImageNet-val) reference points — not gates, for context.
PUBLISHED = {
    "vit_base_patch16_224": 81.8,  # timm model card, ImageNet-1k
    "resnet18": 69.76,  # torchvision ResNet18_Weights.IMAGENET1K_V1 card
}


@torch.no_grad()
def check_vit(device: torch.device, batch_size: int, num_batches: int) -> dict:
    import timm

    from spartan_torch.compat import remap_timm_vit

    ref = timm.create_model("vit_base_patch16_224", pretrained=True).to(device).eval()
    remapped, report = remap_timm_vit({k: v.cpu() for k, v in ref.state_dict().items()})
    assert report.unmatched_source == [], report.unmatched_source
    ours = CompatViT(num_classes=1000).to(device).eval()
    ours.load_state_dict({k: v.to(device) for k, v in remapped.items()}, strict=True)

    torch.manual_seed(0)
    max_diff, min_cos, agree, total = 0.0, 1.0, 0, 0
    for _ in range(num_batches):
        x = torch.randn(batch_size, 3, 224, 224, device=device)
        a, b = ours(x), ref(x)
        d, c = parity_metrics(a, b)
        max_diff = max(max_diff, d)
        min_cos = min(min_cos, c)
        agree += (a.argmax(-1) == b.argmax(-1)).sum().item()
        total += x.size(0)
    return {
        "arch": "ViT-Base/16",
        "source": "timm vit_base_patch16_224 (pretrained)",
        "max_diff": max_diff,
        "cosine": min_cos,
        "pred_agreement": agree / total,
        "published_top1": PUBLISHED["vit_base_patch16_224"],
    }


@torch.no_grad()
def check_resnet18(device: torch.device, batch_size: int, num_batches: int) -> dict:
    import torchvision

    from spartan_torch.compat import remap_torchvision_resnet18

    ref = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    ref = ref.to(device).eval()
    remapped, report = remap_torchvision_resnet18({k: v.cpu() for k, v in ref.state_dict().items()})
    assert report.unmatched_source == [], report.unmatched_source
    ours = CompatResNet18(num_classes=1000).to(device).eval()
    ours.load_state_dict({k: v.to(device) for k, v in remapped.items()}, strict=False)

    torch.manual_seed(0)
    max_diff, min_cos, agree, total = 0.0, 1.0, 0, 0
    for _ in range(num_batches):
        x = torch.randn(batch_size, 3, 224, 224, device=device)
        a, b = ours(x), ref(x)
        d, c = parity_metrics(a, b)
        max_diff = max(max_diff, d)
        min_cos = min(min_cos, c)
        agree += (a.argmax(-1) == b.argmax(-1)).sum().item()
        total += x.size(0)
    return {
        "arch": "ResNet-18",
        "source": "torchvision ResNet18_Weights.IMAGENET1K_V1",
        "max_diff": max_diff,
        "cosine": min_cos,
        "pred_agreement": agree / total,
        "published_top1": PUBLISHED["resnet18"],
    }


def render_table(rows: list[dict]) -> str:
    lines = [
        "| arch | source | max abs diff | cosine | pred agreement | published top-1 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        lines.append(
            f"| {r['arch']} | {r['source']} | {r['max_diff']:.2e} | "
            f"{r['cosine']:.8f} | {r['pred_agreement']:.4f} | {r['published_top1']:.2f} |"
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="all", choices=["all", "vit_base", "resnet18"])
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-batches", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--write-results", action="store_true")
    args = ap.parse_args()

    device = torch.device(args.device)
    rows = []
    if args.arch in ("all", "vit_base"):
        rows.append(check_vit(device, args.batch_size, args.num_batches))
    if args.arch in ("all", "resnet18"):
        rows.append(check_resnet18(device, args.batch_size, args.num_batches))

    table = render_table(rows)
    print(table)

    if args.write_results:
        path = ROOT / "RESULTS.md"
        text = path.read_text() if path.exists() else "# RESULTS\n"
        start, end = "<!-- parity:begin -->", "<!-- parity:end -->"
        block = f"{start}\n{table}\n{end}"
        if start in text and end in text:
            pre, rest = text.split(start, 1)
            _, post = rest.split(end, 1)
            text = pre + block + post
        else:
            text = text.rstrip() + "\n\n" + block + "\n"
        path.write_text(text)
        print(f"updated {path}")


if __name__ == "__main__":
    main()
