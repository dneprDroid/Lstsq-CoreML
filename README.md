# Lstsq-CoreML

CoreML custom layer and converter for [`torch.linalg.lstsq`](https://pytorch.org/docs/stable/generated/torch.linalg.lstsq.html).


## Demo 

Convert [the demo ml-model](converter/demo/convert.py) with `torch.linalg.lstsq` operation to CoreML format:

``` bash 
# Run in the root dir:

python3 -m converter.demo
```
It'll save the ml-model and example input/output tensors to the `DemoApp/generated` directory 
so the demo app can validate the CoreML output results and compare them with the PyTorch output.


### iOS and macOS demo apps

Open `DemoApp/DemoApp.xcodeproj` in Xcode and run the demo app.

The `Test-iOS` target contains the demo for iOS.

The `Test-macOS` target contains the demo for macOS.

In `MLModelTestWorker` it loads the generated CoreML model and the example input tensor from the `DemoApp/generated` directory 
and compares the calculated CoreML output tensor with the PyTorch example output tensor from the `DemoApp/generated` directory:

![Screenshot 2024-06-05 at 14 26 11](https://github.com/dneprDroid/Lstsq-CoreML/assets/13742733/a5ed0da5-9151-47e9-8df5-cc325fb1ab67)


or

in Python:

```python 

A = torch.tensor([
        [1.0,  2.0,  3.0,  4.0,  5.0], 
        [6.0,  7.0,  8.0,  9.0, 10.0],
        [11.0, 12.0, 13.0, 14.0, 15.0]
    ], 
    dtype=torch.float32
)
B = torch.tensor([ 355.0,  930.0, 1505.0 ], dtype=torch.float32)

result = torch.linalg.lstsq(A, B, driver='gelsd') 
print(result)

# torch.return_types.linalg_lstsq(
#     solution=tensor([21.0000, 22.0000, 23.0000, 24.0000, 25.0000]),
#     residuals=tensor([]),
#     rank=tensor(2),
#     singular_values=tensor([3.5127e+01, 2.4654e+00, 1.4961e-06]))

```

in [Swift](https://github.com/dneprDroid/Lstsq-CoreML/blob/af5c47f5795286db401888e73606ac0dc57a1973/demoApp/Test-iOS/Sources/MLModelTestWorker.swift#L56):

```swift 

let a = [
    [1.0,  2.0,  3.0,  4.0,  5.0],
    [6.0,  7.0,  8.0,  9.0, 10.0],
    [11.0, 12.0, 13.0, 14.0, 15.0]
]
let b = [355.0,  930.0, 1505.0]

let testOutput = try demoCoreMLModel.forward(a, b)

print("testOutput: \n\(testOutput)\n")
// testOutput: 
//    solution=[21.000021, 22.00001, 23.000002, 24.0000, 25.0000],
//    rank=2.0,
//    singular_values=[3.5127e+01, 2.465397, 5.823978e-07]

```

## Use in your project


Install this pip package and import it in your converter script:

```bash 

pip install git+https://github.com/dneprDroid/Lstsq-CoreML.git


```

```python
import LstsqConvert

...

# register op so CoreML Tools can find the converter function  
LstsqConvert.register_op()

# and convert your model...
...

```

**NOTE**: In the `coremltools.convert` function you need to set `convert_to="neuralnetwork"`:

```python
mlmodel = coremltools.convert(
    traced_model,
    inputs=...,
    outputs=...,
    convert_to="neuralnetwork"
)
```
#### iOS/macOS app

In your iOS/macOS app add the SwiftPM package from this repository:

```
https://github.com/dneprDroid/Lstsq-CoreML.git
```
CoreML should find and load the custom layers from the `LstsqCoreML` module automatically, so you don't need to do anything.  





