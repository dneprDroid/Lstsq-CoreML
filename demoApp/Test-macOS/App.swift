import Foundation
import AppKit
import CoreML
import LstsqCoreML

@main
class App {
    
    static func main() async {
        do {
            try await test()
        } catch {
            print("error: ", error)
            show(message: "Error", description: error.localizedDescription)
            return
        }
        
        let description = "Output tensors from PyTorch and CoreML custom layers are equal. " +
                          "For more info please check the logs."
        show(
            message: "Success",
            description: description
        )
    }
    
    private static func test() async throws {
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
        
        let (exampleInputTensorA, _) = try NdArrayUtil.readTensor(resource: "example_input_a.json", type: NdArray4d.self)

        let (exampleInputTensorB, _) = try NdArrayUtil.readTensor(resource: "example_input_b.json", type: NdArray3d.self)

        let (_, exampleSolutionOutput) = try NdArrayUtil.readTensor(resource: "example_output_solution.json", type: NdArray3d.self)
        let (_, exampleRankOutput) = try NdArrayUtil.readTensor(resource: "example_output_rank.json", type: NdArray2d.self)
        let (_, exampleSingularValuesOutput) = try NdArrayUtil.readTensor(resource: "example_output_singular_values.json", type: NdArray3d.self)

        let input = Input(a: exampleInputTensorA, b: exampleInputTensorB)
        
        print("loaded")
                
        let outputs = try model.prediction(from: input)
                
        let solutionTensor = outputs.featureValue(for: "solution")?
            .multiArrayValue
        let rankTensor = outputs.featureValue(for: "rank")?
            .multiArrayValue
        let singularValuesTensor = outputs.featureValue(for: "singular_values")?
            .multiArrayValue

        guard let solutionTensor, let rankTensor, let singularValuesTensor else {
            fatalError("output is empty")
        }

        assert(solutionTensor.dataType == .float32)
        assert(rankTensor.dataType == .float32)
        assert(singularValuesTensor.dataType == .float32)

        let solutionArray = solutionTensor.toNdArray3d()
        let rankArray = rankTensor.toNdArray2d()
        let singularValuesArray = singularValuesTensor.toNdArray3d()
        
        print("calculated output (solution): ", solutionArray)
        print("calculated output (rank): ", rankArray)
        print("calculated output (singular_values): ", singularValuesArray)

        let isSolutionValid = NdArrayUtil.validate(
            actual: solutionArray,
            expected: exampleSolutionOutput
        )
        let isRankValid = NdArrayUtil.validate(
            actual: rankArray,
            expected: exampleRankOutput
        )
        let isSingularValuesValid = NdArrayUtil.validate(
            actual: singularValuesArray,
            expected: exampleSingularValuesOutput
        )
        let isValid = isSolutionValid && isRankValid && isSingularValuesValid
        assert(isValid)
    }
    
    static private func show(message: String, description: String) {
        let alert = NSAlert()
        alert.messageText = message
        alert.informativeText = description
        alert.runModal()
    }
}

private extension App {
    
    final class Input: MLFeatureProvider {
        
        let featureNames: Set<String> = ["a", "b"]
        
        private let a: MLMultiArray
        private let b: MLMultiArray
        
        init(a: MLMultiArray, b: MLMultiArray) {
            self.a = a
            self.b = b
        }
        
        func featureValue(for featureName: String) -> MLFeatureValue? {
            switch featureName {
            case "a":
                return MLFeatureValue(multiArray: a)
            case "b":
                return MLFeatureValue(multiArray: b)
            default:
                return .none
            }
        }
    }
}
