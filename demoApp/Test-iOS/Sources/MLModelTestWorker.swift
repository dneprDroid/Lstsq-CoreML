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

        let (_, exampleInputA) = try NdArrayUtil.readTensor(resource: "example_input_a.json", type: NdArray4d.self)

        let (_, exampleInputB) = try NdArrayUtil.readTensor(resource: "example_input_b.json", type: NdArray3d.self)

        let (_, exampleSolutionOutput) = try NdArrayUtil.readTensor(resource: "example_output_solution.json", type: NdArray3d.self)
        let (_, exampleRankOutput) = try NdArrayUtil.readTensor(resource: "example_output_rank.json", type: NdArray2d.self)
        let (_, exampleSingularValuesOutput) = try NdArrayUtil.readTensor(resource: "example_output_singular_values.json", type: NdArray3d.self)
        
        print("loaded")
        
        await onUpdateState(.runningModel)
        
        let output = try model.forward(exampleInputA, exampleInputB)
        
        await onUpdateState(.validation)
        
        let solutionArray = output.solution
        let rankArray = output.rank
        let singularValuesArray = output.singularValues
        
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
        let isOk = isSolutionValid && isRankValid && isSingularValuesValid
        await onUpdateState(.completed(ok: isOk))
    }
}
