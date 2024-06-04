import json

from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.frontend.torch.ops import (
    _get_inputs as mil_get_inputs
)
from coremltools.converters.mil import (
    register_torch_op
)


def register_op():
    register_torch_op(
        _func=torch_lstsq,
        torch_alias=["linalg_lstsq"],
        override=True
    )


def torch_lstsq(context, node):
    inputs = mil_get_inputs(context, node, expected=4)

    a = inputs[0]
    b = inputs[1]
    driver = inputs[3]

    perm = list(range(len(a.shape)))

    m, n = perm[-2], perm[-1]
    perm[-2], perm[-1] = n, m

    a_transposed = mb.transpose(
        x=a,
        perm=perm,
        name=node.name + "_transposed",
    )

    # solution, residuals, rank, singular_values = mb.lstsq_op(
    output = mb.lstsq_op(
        a=a_transposed,
        b=b,
        driver=driver,
        name=node.name
    )

    context.add(output[0], torch_name=node.name)
    context.add(output[1], torch_name=node.outputs[1])
    context.add(output[2], torch_name=node.outputs[2])
    context.add(output[3], torch_name=node.outputs[3])
