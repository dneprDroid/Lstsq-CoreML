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
        # assert self.p1.shape[-1] == self.p2.shape[-2]

        # out_shape = list(self.p2.shape)
        # out_shape[-2] = 1
        # types.dict(
        #     types.str, 
        #     types.tensor
        # )
        # return types.tensor(self.p1.dtype, out_shape)
        def _result(shape):
            return types.tensor(self.a.dtype, shape)
        # return types.dict(
        #     types.str, 
        #     types.tensor(self.a.dtype, [1, 2])
        # )
        # return types.tuple(
        #     (
        #         _result([1,5]),
        #     _result([1]),
        #     _result([1]), 
        #     _result([3])
        #     )
        # ) 
        # return types.list([
        #       _result([1,5]),
        #     _result([1]),
        #     _result([1]), 
        #     _result([3])
        # ])
        dtype = self.a.dtype
        return  types.tensor(dtype, [1, 5]), \
            types.tensor(dtype, [0]), \
            types.tensor(dtype, [1]), \
            types.tensor(dtype, [3])
        # ))