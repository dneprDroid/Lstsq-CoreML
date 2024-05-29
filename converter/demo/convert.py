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
        # result = torch.linalg.lstsq(a, b, driver='gelsd')
        # return result.solution, \
        #     result.residuals, \
        #     result.rank, \
        #     result.singular_values
        gg_solution, residuals, rank, singular_values = torch.linalg.lstsq(a, b, driver='gelsd')
        # print("in forward: ", solution, residuals, rank, singular_values)
        # return solution, \
        #     residuals, \
        #     rank, \
        #     singular_values
        # return solution.flatten() + rank.flatten() + singular_values.flatten()
        # result = torch.zeros([solution.numel()])
        # sizeInfo = torch.zeros(4, 1)
        # sizeInfo[0] = gg_solution.numel()
        # sizeInfo[1] = residuals.numel()
        # sizeInfo[2] = rank.numel()
        # sizeInfo[3] = singular_values.numel()

        return torch.cat([
            # sizeInfo.flatten(),
            
            gg_solution.flatten(), 
            # residuals.flatten(),
            rank.flatten(), 
            singular_values.flatten()
        ])
        # return result



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
"""
inline std::tuple<Tensor, Tensor, Tensor, Tensor> lstsq(
    const Tensor& self,
    const Tensor& b,
    c10::optional<double> cond,
    c10::optional<c10::string_view> driver) {
  return torch::linalg_lstsq(self, b, cond, driver);
}

"""

def convert(output_dir, filename='test-model'):
    torch.return_types.linalg_lstsq

    output_path = os.path.join(output_dir, filename)

    torch_model = TestModel()

    print("generating random input tensor...")
    a = torch.tensor([1,  2,  3,  4,  5, 6,  7,  8,  9, 10, 11, 12, 13, 14, 15], dtype=torch.float32).reshape(3, 5)
    b = torch.tensor([355, 930, 1505], dtype=torch.float32)

    example_input = (a, b)
    example_output = torch_model(a, b)

    print("example output: ", example_output)

    save_as_json((example_input), 'example_input.json', output_dir)
    save_as_json((example_output,), 'example_output.json', output_dir)

    traced_model = torch.jit.trace(torch_model, example_input)

    input_name = "input"
    output_name = "output"

    mil_inputs = [
        coremltools.TensorType(
            name="%s-%i" % (input_name, input_index),
            shape=(x.shape)
        )
        for input_index, x in enumerate(example_input)
    ]
    mil_outputs = [
        coremltools.TensorType(
            name=output_name
        )
    ]
    print("mil_inputs: ", mil_inputs)
    print("mil_outputs: ", mil_outputs)

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
    register_op()

    current_dir = os.path.dirname(os.path.realpath(__file__))
    out_dir_path = os.path.join(current_dir, "../../DemoApp/generated")
    out_dir_path = os.path.abspath(out_dir_path)
    convert(out_dir_path)
