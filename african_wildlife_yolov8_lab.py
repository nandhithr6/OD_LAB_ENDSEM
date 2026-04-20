"""
Advanced Object Detection Lab (YOLOv8) - African Wildlife Dataset

Classes: buffalo, elephant, rhino, zebra

What this script does:
1) Loads a pretrained YOLOv8 model (transfer learning).
2) Fine-tunes it on the provided African Wildlife dataset.
3) Runs inference on test images and visualizes detections.
4) Evaluates using YOLO metrics (Precision, Recall, mAP/IoU-threshold based metrics).
5) Shows optional manual IoU calculation for viva explanation.

Run example:
    python african_wildlife_yolov8_lab.py --data data.yaml --epochs 5 --model yolov8n.pt

Notes:
- Designed for CPU/lab systems: small epochs and lightweight model by default.
- Your dataset should already be in YOLO format with train/val/test splits.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml
from ultralytics import YOLO


AFRICAN_WILDLIFE_ZIP_URL = (
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/african-wildlife.zip"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 African Wildlife Lab")
    parser.add_argument("--data", type=str, default="data.yaml", help="Path to YOLO data.yaml")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Pretrained model weights")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs (5-10 for labs)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--batch", type=int, default=8, help="Batch size (reduce if RAM is low)")
    parser.add_argument("--device", type=str, default="cpu", help="Device: cpu or cuda:0")
    parser.add_argument("--project", type=str, default="runs/wildlife_lab", help="Output project folder")
    parser.add_argument("--name", type=str, default="yolov8n_finetune", help="Run name")
    parser.add_argument("--num_vis", type=int, default=4, help="Number of test images to visualize")
    parser.add_argument(
        "--skip_train",
        action="store_true",
        help="Skip training and only run evaluation + visualization using provided weights.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="",
        help="Path to trained weights (e.g., runs/wildlife_lab/yolov8n_finetune2/weights/best.pt).",
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        default="",
        help="Optional run directory to save outputs when --skip_train is used.",
    )
    parser.add_argument(
        "--show_plots",
        action="store_true",
        help="Display matplotlib windows. Keep OFF for terminal-only lab runs.",
    )
    return parser.parse_args()


def is_valid_african_wildlife_root(root: Path) -> bool:
    return (
        (root / "images" / "train").exists()
        and (root / "images" / "val").exists()
        and (root / "images" / "test").exists()
        and (root / "labels" / "train").exists()
        and (root / "labels" / "val").exists()
        and (root / "labels" / "test").exists()
    )


def find_dataset_root(workspace_dir: Path) -> Path | None:
    # Common locations after extraction.
    candidates = [
        workspace_dir / "african-wildlife",
        workspace_dir,
        workspace_dir / "datasets" / "african-wildlife",
    ]
    for c in candidates:
        if is_valid_african_wildlife_root(c):
            return c

    # Fallback: search one level deep for a folder that matches expected structure.
    for p in workspace_dir.iterdir():
        if p.is_dir() and is_valid_african_wildlife_root(p):
            return p
    return None


def _download_progress(block_num: int, block_size: int, total_size: int) -> None:
    if total_size <= 0:
        return
    downloaded = min(block_num * block_size, total_size)
    percent = (downloaded / total_size) * 100
    print(f"\rDownloading dataset: {percent:6.2f}%", end="")


def download_and_extract_dataset(workspace_dir: Path) -> Path:
    zip_path = workspace_dir / "african-wildlife.zip"
    existing_root = find_dataset_root(workspace_dir)

    if existing_root is not None:
        print(f"Dataset already found: {existing_root}")
        return existing_root

    print("African Wildlife dataset not found locally. Downloading now...")
    urllib.request.urlretrieve(AFRICAN_WILDLIFE_ZIP_URL, zip_path, _download_progress)
    print("\nDownload complete.")

    dataset_root = workspace_dir / "african-wildlife"
    if dataset_root.exists() and not is_valid_african_wildlife_root(dataset_root):
        shutil.rmtree(dataset_root)

    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(workspace_dir)

    detected_root = find_dataset_root(workspace_dir)
    if detected_root is None:
        raise FileNotFoundError(
            "Dataset extracted but expected folder structure was not found. "
            "Expected images/train|val|test and labels/train|val|test."
        )

    print(f"Dataset ready at: {detected_root}")
    return detected_root


def create_african_wildlife_yaml(yaml_path: Path, dataset_root: Path) -> Path:
    yaml_content = {
        "path": str(dataset_root),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "buffalo", 1: "elephant", 2: "rhino", 3: "zebra"},
    }
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(yaml_content, f, sort_keys=False)
    print(f"Created dataset YAML: {yaml_path}")
    return yaml_path


def prepare_data_yaml(data_arg: str) -> str:
    data_path = Path(data_arg)
    if data_path.exists():
        return str(data_path.resolve())

    supported_names = {"data.yaml", "african-wildlife.yaml"}
    if data_path.name not in supported_names:
        raise FileNotFoundError(
            f"'{data_arg}' does not exist. Provide a valid YAML path or use one of {supported_names}."
        )

    workspace_dir = Path.cwd()
    dataset_root = download_and_extract_dataset(workspace_dir)

    if data_path.is_absolute():
        yaml_path = data_path
    else:
        yaml_path = (workspace_dir / data_path).resolve()

    create_african_wildlife_yaml(yaml_path, dataset_root)
    return str(yaml_path)


def explain_transfer_learning() -> None:
    print("\n=== Why pretrained YOLOv8? (Transfer Learning) ===")
    print(
        "A pretrained model (like yolov8n.pt) has already learned useful visual features "
        "(edges, shapes, textures) from large datasets."
    )
    print(
        "During fine-tuning, we reuse that knowledge and adapt it to our 4 wildlife classes "
        "(buffalo, elephant, rhino, zebra)."
    )
    print(
        "This gives faster training, better accuracy with small data, and is ideal for lab "
        "systems with limited CPU/GPU resources."
    )


def load_model(model_path: str) -> YOLO:
    print(f"\nLoading pretrained model: {model_path}")
    model = YOLO(model_path)
    return model


def train_model(model: YOLO, args: argparse.Namespace):
    print("\n=== Training / Fine-tuning ===")
    train_results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=0,
        project=args.project,
        name=args.name,
        verbose=True,
    )
    run_dir = Path(getattr(train_results, "save_dir", Path(args.project) / args.name))
    print(f"Training artifacts directory: {run_dir}")
    return train_results, run_dir


def validate_model(model: YOLO, data_yaml: str, split: str = "test"):
    print("\n=== Validation on test split ===")
    metrics = model.val(data=data_yaml, split=split, device="cpu", workers=0, verbose=False)
    return metrics


def extract_metrics(metrics) -> Dict[str, float]:
    result: Dict[str, float] = {}

    box = getattr(metrics, "box", None)
    if box is None:
        return result

    metric_map = {
        "precision_mean": "mp",
        "recall_mean": "mr",
        "map50": "map50",
        "map75": "map75",
        "map50_95": "map",
    }
    for out_key, attr_name in metric_map.items():
        val = getattr(box, attr_name, None)
        if val is not None:
            result[out_key] = float(val)
    return result


def print_metrics(metrics) -> Dict[str, float]:
    # Ultralytics exposes box metrics on metrics.box in most versions.
    box = getattr(metrics, "box", None)
    metrics_dict = extract_metrics(metrics)

    print("\n=== Evaluation Metrics ===")
    if box is not None:
        if "precision_mean" in metrics_dict:
            print(f"Precision (mean): {metrics_dict['precision_mean']:.4f}")
        if "recall_mean" in metrics_dict:
            print(f"Recall (mean):    {metrics_dict['recall_mean']:.4f}")
        if "map50" in metrics_dict:
            print(f"mAP@0.50:         {metrics_dict['map50']:.4f}")
        if "map75" in metrics_dict:
            print(f"mAP@0.75:         {metrics_dict['map75']:.4f}")
        if "map50_95" in metrics_dict:
            print(f"mAP@0.50:0.95:    {metrics_dict['map50_95']:.4f}")

        print("\nViva note: IoU is used inside mAP computation at thresholds like 0.50 and 0.50:0.95.")
    else:
        print("Could not parse detailed box metrics from this Ultralytics version.")
        print("Raw metrics object:", metrics)

    return metrics_dict


def yolo_xywhn_to_xyxy(
    x_center: float,
    y_center: float,
    w: float,
    h: float,
    img_w: int,
    img_h: int,
) -> np.ndarray:
    x1 = (x_center - w / 2.0) * img_w
    y1 = (y_center - h / 2.0) * img_h
    x2 = (x_center + w / 2.0) * img_w
    y2 = (y_center + h / 2.0) * img_h
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def bbox_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Compute IoU between two boxes in xyxy format."""
    ix1 = max(box1[0], box2[0])
    iy1 = max(box1[1], box2[1])
    ix2 = min(box1[2], box2[2])
    iy2 = min(box1[3], box2[3])

    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])

    union = area1 + area2 - inter_area
    if union <= 0:
        return 0.0
    return float(inter_area / union)


def read_data_yaml(data_yaml: str) -> dict:
    with open(data_yaml, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_split_images_dir(data_cfg: dict, split: str = "test") -> Path:
    """
    Resolves test images directory from YOLO data.yaml.
    Supports both absolute and relative paths.
    """
    root = Path(data_cfg.get("path", ".")).expanduser().resolve()
    split_val = data_cfg.get(split)
    if split_val is None:
        raise FileNotFoundError(f"'{split}' split not found in data.yaml")

    split_path = Path(split_val)
    if split_path.is_absolute():
        return split_path
    return (root / split_path).resolve()


def collect_test_images(test_dir: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = [p for p in test_dir.rglob("*") if p.suffix.lower() in exts]
    return sorted(images)


def visualize_detections(model: YOLO, image_paths: List[Path], out_file: Path, show_plots: bool) -> None:
    if not image_paths:
        print("No test images found for visualization.")
        return

    preds = model.predict([str(p) for p in image_paths], conf=0.25, iou=0.5, device="cpu", verbose=False)

    cols = 2
    rows = int(np.ceil(len(preds) / cols))
    plt.figure(figsize=(12, 5 * rows))

    for i, res in enumerate(preds, start=1):
        plotted_bgr = res.plot()
        plotted_rgb = cv2.cvtColor(plotted_bgr, cv2.COLOR_BGR2RGB)

        plt.subplot(rows, cols, i)
        plt.imshow(plotted_rgb)
        plt.title(f"Detection {i}: {Path(res.path).name}")
        plt.axis("off")

    plt.tight_layout()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=200)
    if show_plots:
        plt.show()
    plt.close()
    print(f"Saved detection visualization: {out_file}")


def show_training_curves(run_dir: Path, show_plots: bool) -> None:
    results_png = run_dir / "results.png"
    if results_png.exists():
        img = cv2.imread(str(results_png))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.figure(figsize=(12, 6))
            plt.imshow(img)
            plt.title("YOLOv8 Training Curves")
            plt.axis("off")
            plt.tight_layout()
            preview_path = run_dir / "training_curves_preview.png"
            plt.savefig(preview_path, dpi=200)
            if show_plots:
                plt.show()
            plt.close()
            print(f"Training curves loaded from: {results_png}")
            print(f"Saved training preview: {preview_path}")
            return
    print("Training curve image not found yet (results.png).")


def manual_iou_demo(model: YOLO, data_yaml: str, max_images: int = 2) -> List[dict]:
    """
    Simple manual IoU demo:
    - Pick a few test images.
    - Compare highest-confidence prediction against GT box of the same class.
    This is for explanation in viva, not a full benchmark.
    """
    print("\n=== Optional Manual IoU Demo ===")
    cfg = read_data_yaml(data_yaml)
    test_img_dir = resolve_split_images_dir(cfg, split="test")
    test_images = collect_test_images(test_img_dir)

    if not test_images:
        print("No test images found for manual IoU demo.")
        return []

    random.shuffle(test_images)
    samples = test_images[:max_images]
    samples_out: List[dict] = []

    for img_path in samples:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        # YOLO label path convention: images/.../x.jpg -> labels/.../x.txt
        label_path = Path(str(img_path).replace("images", "labels"))
        label_path = label_path.with_suffix(".txt")
        if not label_path.exists():
            print(f"Label missing for {img_path.name}, skipping.")
            continue

        gt_boxes: List[Tuple[int, np.ndarray]] = []
        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls_id = int(float(parts[0]))
                xc, yc, bw, bh = map(float, parts[1:])
                box = yolo_xywhn_to_xyxy(xc, yc, bw, bh, w, h)
                gt_boxes.append((cls_id, box))

        if not gt_boxes:
            print(f"No GT boxes in {label_path.name}, skipping.")
            continue

        pred = model.predict(str(img_path), conf=0.25, iou=0.5, device="cpu", verbose=False)[0]
        if pred.boxes is None or len(pred.boxes) == 0:
            print(f"No predictions for {img_path.name}")
            continue

        # Highest-confidence predicted box.
        confs = pred.boxes.conf.cpu().numpy()
        top_idx = int(np.argmax(confs))
        pred_box = pred.boxes.xyxy.cpu().numpy()[top_idx]
        pred_cls = int(pred.boxes.cls.cpu().numpy()[top_idx])

        # Match with GT of same class and compute best IoU.
        same_cls_gts = [b for c, b in gt_boxes if c == pred_cls]
        if not same_cls_gts:
            print(f"{img_path.name}: predicted class has no GT of same class for IoU demo.")
            continue

        ious = [bbox_iou(pred_box, gt_box) for gt_box in same_cls_gts]
        best_iou = max(ious)

        print(
            f"{img_path.name} | Pred class={pred_cls}, conf={confs[top_idx]:.3f}, "
            f"best IoU with same-class GT={best_iou:.3f}"
        )
        samples_out.append(
            {
                "image": img_path.name,
                "pred_class": pred_cls,
                "pred_conf": float(confs[top_idx]),
                "best_iou_same_class_gt": float(best_iou),
            }
        )

    print("Manual IoU formula used:")
    print("IoU = Area(Intersection) / Area(Union)")
    return samples_out


def performance_analysis_for_viva() -> str:
    lines = [
        "=== Exam-Oriented Performance Analysis ===",
        "1) Dense forest background:",
        "- Challenge: busy textures (leaves, branches, shadows) create false positives.",
        "- Impact: precision can drop because model may detect background patterns as animals.",
        "- Why YOLOv8 helps: pretrained features improve object localization in cluttered scenes.",
        "",
        "2) Camouflage conditions:",
        "- Challenge: low contrast between animal and background reduces visibility of edges.",
        "- Impact: recall can drop because some animals are missed (false negatives).",
        "- Typical observation: larger animals are detected better than partially hidden ones.",
        "",
        "Short viva conclusion:",
        "Fine-tuning a pretrained YOLOv8 model gives good baseline performance quickly, "
        "but dense background and camouflage remain hard cases that affect precision/recall.",
    ]
    text = "\n".join(lines)
    print("\n=== Exam-Oriented Performance Analysis ===")
    print("1) Dense forest background:")
    print("- Challenge: busy textures (leaves, branches, shadows) create false positives.")
    print("- Impact: precision can drop because model may detect background patterns as animals.")
    print("- Why YOLOv8 helps: pretrained features improve object localization in cluttered scenes.")

    print("\n2) Camouflage conditions:")
    print("- Challenge: low contrast between animal and background reduces visibility of edges.")
    print("- Impact: recall can drop because some animals are missed (false negatives).")
    print("- Typical observation: larger animals are detected better than partially hidden ones.")

    print("\nShort viva conclusion:")
    print(
        "Fine-tuning a pretrained YOLOv8 model gives good baseline performance quickly, "
        "but dense background and camouflage remain hard cases that affect precision/recall."
    )
    return text


def save_reports(
    run_dir: Path,
    args: argparse.Namespace,
    metrics_dict: Dict[str, float],
    manual_iou_samples: List[dict],
    analysis_text: str,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)

    metrics_txt = run_dir / "evaluation_metrics.txt"
    lines = ["YOLOv8 Evaluation Metrics"]
    if metrics_dict:
        for k, v in metrics_dict.items():
            lines.append(f"{k}: {v:.4f}")
    else:
        lines.append("Metrics unavailable for this run.")
    metrics_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    metrics_json = run_dir / "evaluation_metrics.json"
    metrics_json.write_text(json.dumps(metrics_dict, indent=2), encoding="utf-8")

    iou_json = run_dir / "manual_iou_samples.json"
    iou_json.write_text(json.dumps(manual_iou_samples, indent=2), encoding="utf-8")

    analysis_file = run_dir / "viva_analysis.txt"
    analysis_file.write_text(analysis_text + "\n", encoding="utf-8")

    summary_md = run_dir / "run_summary.md"
    summary_lines = [
        "# African Wildlife YOLOv8 Run Summary",
        "",
        "## Run Configuration",
        f"- Model: {args.model}",
        f"- Data YAML: {args.data}",
        f"- Epochs: {args.epochs}",
        f"- Image size: {args.imgsz}",
        f"- Batch: {args.batch}",
        f"- Device: {args.device}",
        "",
        "## Metrics",
    ]
    if metrics_dict:
        for k, v in metrics_dict.items():
            summary_lines.append(f"- {k}: {v:.4f}")
    else:
        summary_lines.append("- Metrics unavailable for this run.")

    summary_lines.extend(
        [
            "",
            "## Key Files",
            "- results.csv (training log)",
            "- results.png (YOLO training curves)",
            "- training_curves_preview.png (saved training plot)",
            "- test_detections.png (inference visualization)",
            "- evaluation_metrics.txt / .json",
            "- manual_iou_samples.json",
            "- viva_analysis.txt",
        ]
    )
    summary_md.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print("\nSaved report files:")
    print(f"- {metrics_txt}")
    print(f"- {metrics_json}")
    print(f"- {iou_json}")
    print(f"- {analysis_file}")
    print(f"- {summary_md}")


def main() -> None:
    args = parse_args()
    args.data = prepare_data_yaml(args.data)

    explain_transfer_learning()

    if args.skip_train:
        if args.weights:
            weight_path = Path(args.weights).resolve()
            if not weight_path.exists():
                raise FileNotFoundError(f"Weights file not found: {weight_path}")
            print(f"\nSkipping training. Loading trained weights: {weight_path}")
            model = YOLO(str(weight_path))
            if args.run_dir:
                run_dir = Path(args.run_dir).resolve()
            else:
                # Common layout: <run_dir>/weights/best.pt
                run_dir = weight_path.parent.parent if weight_path.parent.name == "weights" else Path.cwd()
        else:
            print("\nSkipping training. Loading model from --model for eval/visualization.")
            model = load_model(args.model)
            run_dir = Path(args.run_dir).resolve() if args.run_dir else Path(args.project) / "inference_only"
    else:
        model = load_model(args.model)

        # 1) Train/fine-tune
        _, run_dir = train_model(model, args)

    # 2) Validate and print metrics
    metrics = validate_model(model, args.data, split="test")
    metrics_dict = print_metrics(metrics)

    # 3) Visualize training curves
    show_training_curves(run_dir, args.show_plots)

    # 4) Inference and detection visualization
    data_cfg = read_data_yaml(args.data)
    test_img_dir = resolve_split_images_dir(data_cfg, split="test")
    test_images = collect_test_images(test_img_dir)

    if len(test_images) == 0:
        print("No test images found. Check dataset paths in data.yaml")
    else:
        chosen = test_images[: args.num_vis]
        vis_path = run_dir / "test_detections.png"
        visualize_detections(model, chosen, vis_path, args.show_plots)

    # 5) Optional manual IoU demonstration
    manual_iou_samples = manual_iou_demo(model, args.data, max_images=2)

    # 6) Exam-oriented analysis statements
    analysis_text = performance_analysis_for_viva()

    # 7) Persist clear output logs/reports in run folder
    save_reports(run_dir, args, metrics_dict, manual_iou_samples, analysis_text)


if __name__ == "__main__":
    main()
