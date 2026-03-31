from typing import Any

import numpy as np
import onnx
import torch
import torch.onnx
from PIL import Image
from transformers import YolosForObjectDetection, YolosImageProcessor
from transformers.image_processing_base import ImageProcessorType
from transformers.modeling_utils import SpecificPreTrainedModelType
import onnxscript

from onnx_edit import add_static_quantization


def get_yolos() -> (SpecificPreTrainedModelType, ImageProcessorType):
    model_url = 'hustvl/yolos-base'
    model = YolosForObjectDetection.from_pretrained(model_url)
    image_processor = YolosImageProcessor.from_pretrained(model_url)

    dummy_image = Image.fromarray(np.random.randint(0, 255, (800, 1333, 3), dtype=np.uint8))
    inputs = image_processor(images=dummy_image, return_tensors="pt")

    return model, inputs['pixel_values']


def convert_pytorch_to_onnx(
        model: Any,
        dummy_input: torch.Tensor,
        onnx_file_path: str
):
    model.eval()

    onnx_program = torch.onnx.export(model, dummy_input, dynamo=True)
    onnx_program.save(f"{onnx_file_path}.onnx")

    print(f"Model successfully converted to ONNX and saved at {onnx_file_path}.onnx")


def check_onnx_model(onnx_file_path):
    onnx_model = onnx.load(f"{onnx_file_path}.onnx")
    onnx.checker.check_model(onnx_model)


def main():
    # model, dummy_input = get_yolos()
    # print(f"Dummy input shape: {dummy_input.shape}")
    #
    # convert_pytorch_to_onnx(model, dummy_input, "out_yolos_model")

    onnx_model = onnx.load("out_yolos_model.onnx")
    add_static_quantization(onnx_model, "out_yolos_model_quant")
    check_onnx_model("out_yolos_model_quant")


if __name__ == "__main__":
    main()
