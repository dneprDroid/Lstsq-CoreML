from coremltools.converters.mil.mil.ops.defs._op_reqs import register_op
from coremltools.converters.mil.mil import (
    Operation,
    types
)
from coremltools.converters.mil.mil.input_type import (
    InputSpec,
    TensorInputType,
)


def _process_class_name(classname):
    return 'dneprDroid_' + classname


@register_op(is_custom_op=True)
class lstsq_op(Operation):

    input_spec = InputSpec(
        a=TensorInputType(type_domain="T"),
        b=TensorInputType(type_domain="T"),
        driver=TensorInputType(const=True, type_domain=types.str),
    )

    type_domains = {
        "T": (types.fp32,),
    }

    bindings = {
        "class_name": _process_class_name('lstsq'),
        "input_order": ["a", "b"],
        "parameters": [
            "driver"
        ],
        "description": "lstsq developed by dneprDroid",
    }

    def __init__(self, **kwargs):
        super(lstsq_op, self).__init__(**kwargs)

    def type_inference(self):
        a_shape = self.a.shape
        dtype = self.a.dtype

        batches = list(a_shape[:-2])
        solution_shape = batches + [a_shape[-2]]
        sing_values_shape = batches + [a_shape[-1]]
        rank_shape = [1] if len(batches) == 0 else batches

        return types.tensor(dtype, solution_shape),        \
            types.tensor(dtype, [1]),                      \
            types.tensor(dtype, rank_shape),               \
            types.tensor(dtype, sing_values_shape)
