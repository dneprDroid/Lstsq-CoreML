import Foundation
import SwiftUI
import CoreML
import LstsqCoreML

enum State {
    case initial
    case loadingModel
    case loadingExampleTensors
    case runningModel
    case validation
    case completed(ok: Bool)
}

final class MLModelTestWorker {
    
    var onUpdateState: (State) async -> Void = { _ in }

    func test() async throws {
        await onUpdateState(.loadingModel)
        
        guard let modelUrl = Bundle.main.url(forResource: "test-model.mlmodel.pb", withExtension: nil) else {
            fatalError("Can't find ML model")
        }
        print("loading model....")
        let compiledUrl = try MLModel.compileModel(at: modelUrl)
        defer {
            try? FileManager.default.removeItem(at: compiledUrl)
        }

        let configuration = MLModelConfiguration()
        configuration.computeUnits = .cpuOnly
        
        configuration.allowLowPrecisionAccumulationOnGPU = false

        let model = try MLModel(contentsOf: compiledUrl, configuration: configuration)

        print("loading example inputs/outputs from JSON files...")
        
        await onUpdateState(.loadingExampleTensors)

        let (exampleInputTensorA, _) = try NdArrayUtil.readTensor(resource: "example_input_a.json", type: NdArray4d.self)

        let (exampleInputTensorB, _) = try NdArrayUtil.readTensor(resource: "example_input_b.json", type: NdArray3d.self)

        let (_, exampleOutputArray) = try NdArrayUtil.readTensor(resource: "example_output.json", type: NdArray2d.self)

        let input = Input(a: exampleInputTensorA, b: exampleInputTensorB)
        
        print("loaded")
        
        await onUpdateState(.runningModel)
        
        let output = try model.prediction(from: input)
            .featureValue(for: "output")?
            .multiArrayValue
        
        await onUpdateState(.validation)

        guard let output else { fatalError("output is empty") }

        assert(output.dataType == .float32)
        
        let flattenArray = output.toFlattenArray(for: Float32.self)
        print("calculated output (flatten tensor): ", flattenArray)
        
        let isOk = NdArrayUtil.validate(
            actual: .init(value: [flattenArray]),
            expected: exampleOutputArray
        )
        await onUpdateState(.completed(ok: isOk))
    }
}
    
extension MLModelTestWorker {

    final class Input: MLFeatureProvider {
        
        let featureNames: Set<String> = ["input_0", "input_1"]
        
        private let a: MLMultiArray
        private let b: MLMultiArray
        
        init(a: MLMultiArray, b: MLMultiArray) {
            self.a = a
            self.b = b
        }
        
        func featureValue(for featureName: String) -> MLFeatureValue? {
            switch featureName {
            case "input_0":
                return MLFeatureValue(multiArray: a)
            case "input_1":
                return MLFeatureValue(multiArray: b)
            default:
                return .none
            }
        }
    }
}
