from typing import Any

import onnx
import onnx_graphsurgeon as gs
import numpy as np
from onnx import ModelProto
from onnx_graphsurgeon import Graph

# Const parameters for all quantization nodes
scale = np.array([0.01], dtype=np.float32)
zero_point = np.array([0], dtype=np.int8)


def add_quant_and_dequant_to_matmul_node(
        model_graph: Graph,
        matmul_node: Any
):
    matmul_node_name = matmul_node.name
    matmul_output = matmul_node.outputs[0]
    matmul_output_name = matmul_output.name

    for i, matmul_input in enumerate(matmul_node.inputs):
        if isinstance(matmul_input, gs.Constant):  # weights
            continue

        matmul_input_name = matmul_input.name

        quant_input = gs.Variable(
            name=f"{matmul_input_name}_Quant",
            dtype=np.int8
        )
        quant_node = gs.Node(
            op="QuantizeLinear",
            name=f"{matmul_input_name}_Quant_Input_{i}",
            inputs=[
                matmul_input,
                gs.Constant(f"{matmul_node_name}_Scale_{i}", values=scale),
                gs.Constant(f"{matmul_node_name}_Zero_Point_{i}", values=zero_point)
            ],
            outputs=[quant_input]
        )

        # Redirect matmul input tensor
        matmul_node.inputs[i] = quant_input

        model_graph.nodes.extend([quant_node])

    dequant_output = gs.Variable(
        name=f"{matmul_output_name}_Dequant",
        dtype=np.float32
    )
    dequant_node = gs.Node(
        op="DequantizeLinear",
        name=f"{matmul_node_name}_Dequant_Output",
        inputs=[
            matmul_output,
            gs.Constant(f"{matmul_node_name}_Scale", values=scale),
            gs.Constant(f"{matmul_node_name}_Zero_Point", values=zero_point)
        ],
        outputs=[dequant_output]
    )

    model_graph.nodes.extend([dequant_node])

    # Redirect matmul output tensor
    matmul_node.outputs[0] = dequant_node.inputs[0]

    # Redirect matmul input for all users of current matmul node
    for out in matmul_node.outputs:
        for user_node in out.outputs:
            for i, inp in enumerate(user_node.inputs):
                if inp == out:  # link to old output tensor
                    user_node.inputs[i] = dequant_output


def add_static_quantization(onnx_model: ModelProto, quant_onnx_filename: str):
    model_graph = gs.import_onnx(onnx_model)
    matmul_nodes = [node for node in model_graph.nodes if node.op == "MatMul"]

    for matmul_node in matmul_nodes:
        add_quant_and_dequant_to_matmul_node(model_graph, matmul_node)

    model_graph.cleanup().toposort()
    onnx.save(gs.export_onnx(model_graph), f"{quant_onnx_filename}.onnx")
