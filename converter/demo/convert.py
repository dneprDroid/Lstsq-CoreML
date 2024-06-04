"""
Run in the root dir:

python3 -m converter.demo

"""
import os
import shutil
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.ops.deform_conv import deform_conv2d

import coremltools

from .. import register_op


class TestModel(nn.Module):

    def forward(self, a, b):
        solution, residuals, rank, singular_values = torch.linalg.lstsq(
            a, b, driver='gelsd')

        return solution, rank, singular_values

def rm(path):
    if not os.path.exists(path):
        return
    if os.path.isdir(path):
        shutil.rmtree(path)
    else:
        os.remove(path)


def save_as_json(tensors, filename, output_dir):
    values = [tensor.numpy().tolist() for tensor in tensors]
    values_str = json.dumps(values)

    path = os.path.join(output_dir, filename)
    with open(path, 'w') as file:
        file.write(values_str)


def convert(output_dir, filename='test-model'):
    register_op()

    output_path = os.path.join(output_dir, filename)

    torch_model = TestModel()

    print("generating random input tensors...")

    a = torch.rand(2, 4, 3, 5).type(torch.float32)
    b = torch.rand(2, 4, 3).type(torch.float32)

    example_input = (a, b)
    example_output = torch_model(a, b)

    print("example output: ", [out.shape for out in example_output], example_output)

    save_as_json(a, 'example_input_a.json', output_dir)
    save_as_json(b, 'example_input_b.json', output_dir)

    solution, rank, singular_values = example_output
    save_as_json(solution, 'example_output_solution.json', output_dir)
    save_as_json(rank, 'example_output_rank.json', output_dir)
    save_as_json(singular_values, 'example_output_singular_values.json', output_dir)

    traced_model = torch.jit.trace(torch_model, example_input)

    input_names = ["a", "b"]
    output_names = ["solution", "rank", "singular_values"]

    mil_inputs = [
        coremltools.TensorType(
            name=input_names[input_index],
            shape=(x.shape)
        )
        for input_index, x in enumerate(example_input)
    ]
    mil_outputs = [
        coremltools.TensorType(
            name=output_names[out_index]
        )
        for out_index, x in enumerate(example_output)
    ]

    mlmodel = coremltools.convert(
        traced_model,
        inputs=mil_inputs,
        outputs=mil_outputs,
        convert_to="neuralnetwork"
    )
    mlmodel_path = output_path + ".mlmodel"

    out_pb_path = mlmodel_path + ".pb"

    rm(mlmodel_path)
    rm(output_path)
    rm(out_pb_path)

    mlmodel.save(mlmodel_path)

    shutil.copyfile(mlmodel_path, out_pb_path)

    print(f"Saved to {output_dir}")


def main():
    current_dir = os.path.dirname(os.path.realpath(__file__))
    out_dir_path = os.path.join(current_dir, "../../demoApp/generated")
    out_dir_path = os.path.abspath(out_dir_path)
    convert(out_dir_path)
