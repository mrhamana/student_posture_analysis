"""
Model Server for Student Posture Analysis

This server loads your .pt model file and exposes a /predict endpoint.
The main backend sends images/videos here for inference.

Usage:
    1. Place your .pt file in model_server/models/posture_model.pt
    2. Run: cd model_server && python server.py
    3. The backend will automatically connect to http://localhost:8001/predict
"""

import io
import os
import json
import logging
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional

import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.responses import StreamingResponse
import uvicorn
import torch
import torch.nn.functional as F

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Posture Model Server", version="1.0.0")

# ─── Model Loading ─────────────────────────────────────────────────────────────

MODEL_DIR = Path(__file__).parent / "models"
MODEL_PATH = os.environ.get("MODEL_PATH", "")
MODEL_PATHS = os.environ.get("MODEL_PATHS", "")
GNN_MODEL_PATH = os.environ.get(
    "GNN_MODEL_PATH", str(MODEL_DIR / "gnn_posture.pt")
)

# The 8 posture classes the model outputs
POSTURE_CLASSES = [
    "Listening",
    "Looking Screen",
    "Hand Raising",
    "Sleeping / Head Down",
    "Turning Back",
    "Standing",
    "Writing",
    "Reading",
]

models: Dict[str, Any] = {}
model_infos: Dict[str, Dict[str, Any]] = {}  # Extracted model metadata
default_model_name: Optional[str] = None
gnn_model: Optional[torch.nn.Module] = None
gnn_info: Dict[str, Any] = {}
gnn_device: str = "cpu"

try:
    from torch_geometric.nn import GATConv
    HAS_PYG = True
except Exception:
    HAS_PYG = False


class GNNClassifier(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, heads: int = 4):
        super().__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads, concat=True)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=1, concat=False)
        self.lin = torch.nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        return self.lin(x)


def _extract_model_info(yolo_model, model_path: str) -> Dict[str, Any]:
    """Extract detailed metadata from a loaded YOLO model."""
    info: Dict[str, Any] = {}
    try:
        nn_model = yolo_model.model  # The underlying torch nn.Module

        # Basic identity
        info["model_path"] = model_path
        info["task"] = getattr(yolo_model, "task", "detect")
        info["model_name"] = getattr(yolo_model, "ckpt_path", model_path).split("/")[-1].split("\\")[-1]

        # Class names
        names = yolo_model.names  # dict {0: 'Listening', 1: 'Looking Screen', ...}
        info["num_classes"] = len(names)
        info["class_names"] = names

        # Parameter counts
        total_params = sum(p.numel() for p in nn_model.parameters())
        trainable_params = sum(p.numel() for p in nn_model.parameters() if p.requires_grad)
        info["total_parameters"] = total_params
        info["trainable_parameters"] = trainable_params
        info["total_parameters_millions"] = round(total_params / 1e6, 2)

        # Model architecture YAML (if available)
        if hasattr(nn_model, "yaml"):
            info["architecture"] = nn_model.yaml
        if hasattr(nn_model, "yaml_file"):
            info["architecture_variant"] = nn_model.yaml_file

        # Input image size
        if hasattr(yolo_model, "overrides"):
            info["input_size"] = yolo_model.overrides.get("imgsz", 640)

        # Layer info
        layer_types: Dict[str, int] = {}
        total_layers = 0
        for module in nn_model.modules():
            layer_name = type(module).__name__
            layer_types[layer_name] = layer_types.get(layer_name, 0) + 1
            total_layers += 1
        info["total_layers"] = total_layers
        info["layer_types"] = layer_types

        # GFLOPs and speed (if available from training results)
        if hasattr(nn_model, "info"):
            try:
                # This returns a tuple (layers, params, gradients, flops)
                info_tuple = nn_model.info(verbose=False)
                if isinstance(info_tuple, tuple) and len(info_tuple) >= 4:
                    info["gflops"] = round(info_tuple[3], 2)
            except Exception:
                pass

        # Training metadata (if saved in checkpoint)
        if hasattr(yolo_model, "ckpt") and yolo_model.ckpt:
            ckpt = yolo_model.ckpt
            if isinstance(ckpt, dict):
                if "epoch" in ckpt:
                    info["trained_epochs"] = ckpt["epoch"]
                if "best_fitness" in ckpt:
                    info["best_fitness"] = round(float(ckpt["best_fitness"]), 4)
                if "date" in ckpt:
                    info["training_date"] = ckpt["date"]
                # Training args
                train_args = ckpt.get("train_args", {})
                if train_args:
                    info["training_config"] = {
                        k: v for k, v in train_args.items()
                        if k in (
                            "epochs", "batch", "imgsz", "optimizer", "lr0", "lrf",
                            "momentum", "weight_decay", "data", "device", "workers",
                            "patience", "augment", "model",
                        )
                    }

        # File size
        if os.path.exists(model_path):
            size_bytes = os.path.getsize(model_path)
            info["file_size_mb"] = round(size_bytes / (1024 * 1024), 2)

    except Exception as e:
        logger.warning(f"Could not extract some model info: {e}")
        info["extraction_warning"] = str(e)

    return info


def _get_candidate_model_paths() -> List[str]:
    paths: List[str] = []

    if MODEL_PATHS:
        for p in MODEL_PATHS.split(os.pathsep):
            p = p.strip()
            if p:
                paths.append(p)

    if MODEL_PATH:
        paths.append(MODEL_PATH)

    # Auto-discover models in the models folder
    for p in sorted(MODEL_DIR.glob("*.pt")):
        paths.append(str(p))

    # De-duplicate while preserving order
    unique_paths: List[str] = []
    seen = set()
    for p in paths:
        if p not in seen:
            unique_paths.append(p)
            seen.add(p)
    return unique_paths


def load_models():
    """Load YOLO models from the .pt files."""
    global models, model_infos, default_model_name
    try:
        from ultralytics import YOLO

        models = {}
        model_infos = {}
        default_model_name = None

        candidate_paths = _get_candidate_model_paths()
        loaded_any = False

        for model_path in candidate_paths:
            if not os.path.exists(model_path):
                continue
            try:
                yolo_model = YOLO(model_path)
                model_name = Path(model_path).name
                models[model_name] = yolo_model
                model_infos[model_name] = _extract_model_info(yolo_model, model_path)
                loaded_any = True
                if default_model_name is None:
                    default_model_name = model_name
                logger.info(f"Model loaded successfully from {model_path}")
            except Exception as e:
                logger.error(f"Failed to load model at {model_path}: {e}")

        if not loaded_any:
            logger.warning("No model files found. Using demo mode (random detections).")
            models = {}
            model_infos = {}
            default_model_name = None
    except Exception as e:
        logger.error(f"Failed to load models: {e}. Running in demo mode.")
        models = {}
        model_infos = {}
        default_model_name = None


def _load_gnn_model():
    global gnn_model, gnn_info, gnn_device

    gnn_model = None
    gnn_info = {}

    if not HAS_PYG:
        logger.warning("torch_geometric not available; GNN refinement disabled.")
        return

    if not os.path.exists(GNN_MODEL_PATH):
        logger.info("No GNN model found at %s; GNN refinement disabled.", GNN_MODEL_PATH)
        return

    gnn_device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        checkpoint = torch.load(GNN_MODEL_PATH, map_location=gnn_device)

        if isinstance(checkpoint, dict):
            state_dict = (
                checkpoint.get("model_state_dict")
                or checkpoint.get("state_dict")
                or checkpoint
            )
            input_dim = int(checkpoint.get("input_dim", 7 + len(POSTURE_CLASSES)))
            num_classes = int(checkpoint.get("num_classes", len(POSTURE_CLASSES)))
            hidden_dim = int(checkpoint.get("hidden_dim", 64))
        else:
            raise ValueError("Unsupported GNN checkpoint format")

        model = GNNClassifier(input_dim=input_dim, hidden_dim=hidden_dim, num_classes=num_classes)
        model.load_state_dict(state_dict, strict=False)
        model.to(gnn_device)
        model.eval()

        gnn_model = model
        gnn_info = {
            "model_path": GNN_MODEL_PATH,
            "input_dim": input_dim,
            "num_classes": num_classes,
            "hidden_dim": hidden_dim,
            "device": gnn_device,
        }
        logger.info("GNN model loaded successfully from %s", GNN_MODEL_PATH)
    except Exception as e:
        logger.error("Failed to load GNN model: %s", e)
        gnn_model = None
        gnn_info = {}


def _list_model_files() -> List[str]:
    return sorted([p.name for p in MODEL_DIR.glob("*.pt")])


# ─── Demo Mode (when no .pt file is available) ────────────────────────────────


def _generate_demo_detections(
    frame_id: int, width: int = 1280, height: int = 720
) -> Dict[str, Any]:
    """Generate realistic-looking demo detections for testing without a model."""
    import random

    random.seed(frame_id * 7)  # Deterministic per frame for consistency

    num_students = random.randint(5, 15)
    students = []

    for i in range(num_students):
        # Generate non-overlapping bounding boxes
        x1 = random.randint(50, width - 200)
        y1 = random.randint(50, height - 250)
        box_w = random.randint(80, 160)
        box_h = random.randint(150, 250)
        x2 = min(x1 + box_w, width - 10)
        y2 = min(y1 + box_h, height - 10)

        posture = random.choice(POSTURE_CLASSES)
        confidence = round(random.uniform(0.65, 0.99), 4)

        students.append(
            {
                "id": i + 1,
                "bbox": [x1, y1, x2, y2],
                "posture": posture,
                "confidence": confidence,
            }
        )

    return {"frame_id": frame_id, "students": students}


# ─── Real Inference ────────────────────────────────────────────────────────────


def _resolve_model_name(requested: Optional[str]) -> Optional[str]:
    if requested and requested in models:
        return requested
    return default_model_name


def _run_inference_on_frame(
    frame: np.ndarray,
    frame_id: int,
    requested_model: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the actual YOLO model on a single frame."""
    model_name = _resolve_model_name(requested_model)
    if model_name is None:
        h, w = frame.shape[:2]
        return _generate_demo_detections(frame_id, w, h)

    yolo_model = models[model_name]
    results = yolo_model.track(frame, persist=True, verbose=False)
    students = []

    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue

        for i in range(len(boxes)):
            bbox = boxes.xyxy[i].cpu().numpy().tolist()
            conf = float(boxes.conf[i].cpu().numpy())
            cls_id = int(boxes.cls[i].cpu().numpy())

            # Get tracker ID (BoT-SORT)
            tracker_id = (
                int(boxes.id[i].cpu().numpy()) if boxes.id is not None else i + 1
            )

            # Map class index to posture label
            posture = (
                POSTURE_CLASSES[cls_id]
                if cls_id < len(POSTURE_CLASSES)
                else "Listening"
            )

            students.append(
                {
                    "id": tracker_id,
                    "bbox": [round(b, 1) for b in bbox],
                    "posture": posture,
                    "confidence": round(conf, 4),
                    "_class_id": cls_id,
                }
            )

    if gnn_model is not None:
        students = _apply_gnn_refinement(students, frame.shape)

    for student in students:
        student.pop("_class_id", None)

    return {"frame_id": frame_id, "students": students, "model": model_name}


def _build_gnn_features(
    students: List[Dict[str, Any]], frame_shape: tuple
) -> torch.Tensor:
    height, width = frame_shape[:2]
    num_classes = len(POSTURE_CLASSES)
    expected_dim = int(gnn_info.get("input_dim", 7 + num_classes))

    features = []
    for student in students:
        x1, y1, x2, y2 = student["bbox"]
        conf = float(student["confidence"])
        cls_id = int(student.get("_class_id", 0))

        nx1 = x1 / max(width, 1)
        ny1 = y1 / max(height, 1)
        nx2 = x2 / max(width, 1)
        ny2 = y2 / max(height, 1)
        w = max(nx2 - nx1, 0.0)
        h = max(ny2 - ny1, 0.0)

        one_hot = [0.0] * num_classes
        if 0 <= cls_id < num_classes:
            one_hot[cls_id] = 1.0

        # Feature vector: bbox geometry + conf + class one-hot
        features.append([nx1, ny1, nx2, ny2, w, h, conf] + one_hot)

    x = torch.tensor(features, dtype=torch.float32, device=gnn_device)
    if x.size(1) < expected_dim:
        pad = torch.zeros((x.size(0), expected_dim - x.size(1)), device=gnn_device)
        x = torch.cat([x, pad], dim=1)
    elif x.size(1) > expected_dim:
        x = x[:, :expected_dim]
    return x


def _build_edge_index(centers: torch.Tensor, k: int = 5) -> torch.Tensor:
    num_nodes = centers.size(0)
    if num_nodes < 2:
        return torch.empty((2, 0), dtype=torch.long, device=centers.device)

    dist = torch.cdist(centers, centers)
    k = min(k, num_nodes - 1)
    knn = dist.argsort(dim=1)[:, 1 : k + 1]

    row = torch.arange(num_nodes, device=centers.device).unsqueeze(1).repeat(1, k).reshape(-1)
    col = knn.reshape(-1)

    edge_index = torch.stack([torch.cat([row, col]), torch.cat([col, row])], dim=0)
    return edge_index


def _apply_gnn_refinement(
    students: List[Dict[str, Any]], frame_shape: tuple
) -> List[Dict[str, Any]]:
    if gnn_model is None or len(students) < 2:
        return students

    try:
        x = _build_gnn_features(students, frame_shape)
        centers = x[:, 0:2] + (x[:, 2:4] - x[:, 0:2]) / 2.0
        edge_index = _build_edge_index(centers)

        with torch.no_grad():
            logits = gnn_model(x, edge_index)
            probs = torch.softmax(logits, dim=1)
            gnn_conf, gnn_cls = probs.max(dim=1)

        for idx, student in enumerate(students):
            new_conf = float(gnn_conf[idx].item())
            if new_conf >= float(student.get("confidence", 0.0)):
                cls_id = int(gnn_cls[idx].item())
                if 0 <= cls_id < len(POSTURE_CLASSES):
                    student["posture"] = POSTURE_CLASSES[cls_id]
                    student["confidence"] = round(new_conf, 4)

    except Exception as e:
        logger.warning("GNN refinement failed: %s", e)

    return students


# ─── Endpoints ─────────────────────────────────────────────────────────────────


@app.on_event("startup")
def startup():
    load_models()
    _load_gnn_model()


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": len(models) > 0,
        "mode": "inference" if len(models) > 0 else "demo",
        "default_model": default_model_name,
        "available_models": sorted(models.keys()),
        "model_dir": str(MODEL_DIR),
        "discovered_models": _list_model_files(),
        "gnn_loaded": gnn_model is not None,
        "gnn_info": gnn_info or None,
    }


@app.get("/model-info")
def get_model_info():
    """
    Return detailed information extracted from the loaded YOLO model.
    Includes: architecture, parameter counts, class names, layer breakdown,
    training config, file size, and more.
    """
    if len(models) == 0:
        load_models()
    if len(models) == 0:
        return {
            "mode": "demo",
            "message": "No model loaded. Running in demo mode with random detections.",
            "posture_classes": POSTURE_CLASSES,
            "model_dir": str(MODEL_DIR),
            "available_models": _list_model_files(),
        }
    return {
        "mode": "inference",
        "default_model": default_model_name,
        "available_models": sorted(models.keys()),
        "info": model_infos,
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    mode: str = Query("image", regex="^(image|video)$"),
    model: Optional[str] = Query(None),
):
    """
    Run posture detection on an image or video.

    - Image mode: returns a single JSON object
    - Video mode: returns newline-delimited JSON (one line per frame)
    """
    contents = await file.read()

    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Empty file")

    if len(models) == 0:
        load_models()

    if mode == "image":
        return await _process_image(contents, model)
    else:
        return StreamingResponse(
            _process_video_stream(contents, model),
            media_type="application/x-ndjson",
        )


async def _process_image(contents: bytes, requested_model: Optional[str]) -> Dict[str, Any]:
    """Process a single image."""
    try:
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Could not decode image")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    result = _run_inference_on_frame(frame, frame_id=0, requested_model=requested_model)
    return result


async def _process_video_stream(contents: bytes, requested_model: Optional[str]):
    """Process video frame by frame and yield NDJSON."""
    # Write to temp file (OpenCV needs a file path)
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            yield json.dumps({"error": "Could not open video"}) + "\n"
            return

        frame_id = 0
        # Process every Nth frame for efficiency (skip_frames=1 means every frame)
        skip_frames = int(os.environ.get("SKIP_FRAMES", "5"))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_id % skip_frames == 0:
                result = _run_inference_on_frame(
                    frame,
                    frame_id,
                    requested_model=requested_model,
                )
                yield json.dumps(result) + "\n"

            frame_id += 1

        cap.release()
    finally:
        os.unlink(tmp_path)


# ─── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8001"))
    uvicorn.run(app, host="0.0.0.0", port=port)
