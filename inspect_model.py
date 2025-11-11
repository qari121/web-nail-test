import argparse
import os
from typing import Any, Dict, List

import numpy as np
# Suppress plugin loading errors - set before import
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
# Try to prevent plugin directory loading
if 'TF_PLUGIN_DIR' not in os.environ:
    # Set to empty/non-existent to skip plugin loading
    import tempfile
    os.environ['TF_PLUGIN_DIR'] = tempfile.mkdtemp()
    
import warnings
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
except Exception as e:  # pragma: no cover
    # If import fails due to plugin issues, we can still use TFLite runtime directly
    # But for now, let's raise a clearer error
    import sys
    if 'libmetal_plugin' in str(e) or 'tensorflow-plugins' in str(e):
        print("Warning: TensorFlow plugin loading issue detected.", file=sys.stderr)
        print("This shouldn't affect TFLite inference, but may cause issues.", file=sys.stderr)
        print("Consider removing system-wide TensorFlow plugins if this persists.", file=sys.stderr)
        # Try to continue anyway - TFLite might still work
        raise SystemExit("TensorFlow import failed. Try: pip uninstall tensorflow-plugins (if installed system-wide)") from e
    raise SystemExit("TensorFlow is required. Please install it: pip install 'tensorflow>=2.12.0,<3.0.0'") from e


def _fmt_q(params: Dict[str, Any]) -> str:
    if not params:
        return "-"
    scale = params.get("scale", None)
    zero_point = params.get("zero_point", None)
    return f"scale={scale}, zero_point={zero_point}"


def inspect_tflite_model(model_path: str) -> None:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("=== TFLite Model Inspection ===")
    print(f"Model path: {model_path}")
    print("\n-- Inputs --")
    for i, inp in enumerate(input_details):
        name = inp.get("name")
        shape = inp.get("shape")
        dtype = inp.get("dtype")
        quant = inp.get("quantization_parameters", {})
        print(f"[{i}] name={name}")
        print(f"    shape={tuple(int(x) for x in shape)} dtype={dtype}")
        print(f"    quantization={_fmt_q(quant)}")

    print("\n-- Outputs --")
    for i, out in enumerate(output_details):
        name = out.get("name")
        shape = out.get("shape")
        dtype = out.get("dtype")
        quant = out.get("quantization_parameters", {})
        print(f"[{i}] name={name}")
        print(f"    shape={tuple(int(x) for x in shape)} dtype={dtype}")
        print(f"    quantization={_fmt_q(quant)}")

    # Heuristics to identify detection and prototype outputs
    det_idx = None
    proto_idx = None
    P = None
    outputs = []
    for od in output_details:
        # We only need shapes; don't read tensors for speed
        outputs.append({"shape": tuple(int(x) for x in od["shape"]), "dtype": od["dtype"]})

    # Try to infer proto first
    for i, o in enumerate(outputs):
        shape = o["shape"]
        if len(shape) == 4 and 1 <= shape[-1] <= 256:
            proto_idx = i
            P = shape[-1]
            break
        if len(shape) == 3 and 1 <= shape[-1] <= 256 and shape[0] <= 1024:
            proto_idx = i
            P = shape[-1]
            break

    if P is not None:
        for i, o in enumerate(outputs):
            if i == proto_idx:
                continue
            shape = o["shape"]
            if len(shape) >= 2 and shape[-1] == 5 + P:
                det_idx = i
                break

    # Fallback detection guess
    if det_idx is None:
        for i, o in enumerate(outputs):
            if i == proto_idx:
                continue
            shape = o["shape"]
            if len(shape) >= 2 and shape[-1] >= 6:
                det_idx = i
                break

    print("\n-- Inferred Roles --")
    print(f"proto_idx={proto_idx} (P={P})")
    print(f"det_idx={det_idx}")

    # Summarize expectation check
    expected_input_hw = None
    if input_details and len(input_details[0]["shape"]) == 4:
        expected_input_hw = (int(input_details[0]["shape"][1]), int(input_details[0]["shape"][2]))
    print(f"\nInput HxW (from model): {expected_input_hw}")

    # If present, print quick notes about expected shapes
    if proto_idx is not None:
        s = outputs[proto_idx]["shape"]
        print(f"Proto shape: {s}")
    if det_idx is not None:
        s = outputs[det_idx]["shape"]
        print(f"Detections shape: {s}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a TFLite model's IO details")
    parser.add_argument(
        "--model",
        default="/Users/qari/Desktop/Farhan AI/nails_seg_s_yolov8_v1_float16.tflite",
        help="Path to .tflite model",
    )
    args = parser.parse_args()
    inspect_tflite_model(args.model)


if __name__ == "__main__":
    main()


