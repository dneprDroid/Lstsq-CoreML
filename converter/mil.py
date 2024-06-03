import json

from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.frontend.torch.ops import (
    _get_inputs as mil_get_inputs
)
from coremltools.converters.mil import (
    register_torch_op
)


class _State:
    _view_index = 0
    _flatten_index = 0


def register_op():
    register_torch_op(
        _func=torch_lstsq,
        torch_alias=["linalg_lstsq"],
        override=True
    )


def torch_lstsq(context, node):
    print(f"torch_lstsq - node: '{node.name}'", node.outputs) # dir(node))
    inputs = mil_get_inputs(context, node, expected=4)

    a = inputs[0]
    b = inputs[1]
    driver = inputs[3]

    perm = list(range(len(a.shape)))

    m, n = perm[-2], perm[-1]
    perm[-2], perm[-1] = n, m 

    print('perm: ', perm)
    a = mb.transpose(
        x=a,
        perm=perm,
    )

    # solution, residuals, rank, singular_values = mb.lstsq_op(
    output = mb.lstsq_op(
        a=a,
        b=b,
        driver=driver,
        name=node.name
    )
    """
    solution=tensor([21.0000, 22.0000, 23.0000, 24.0000, 25.0000]),
    residuals=tensor([]),
    rank=tensor(2),
    singular_values=tensor([3.5127e+01, 2.4654e+00, 1.4961e-06]))
    """
    # context.add(output) #, torch_name=node.name)
    # context.add(output, torch_name=node.name)


    context.add(output[0], torch_name=node.name)
    context.add(output[1], torch_name=node.outputs[1])
    context.add(output[2], torch_name=node.outputs[2])
    context.add(output[3], torch_name=node.outputs[3])

    # context.add(solution, torch_name=node.name+'-solution')
    # context.add(residuals, torch_name=node.name+'-residuals')
    # context.add(rank, torch_name=node.name+'-rank')
    # context.add(singular_values, torch_name=node.name+'-singular_values')
