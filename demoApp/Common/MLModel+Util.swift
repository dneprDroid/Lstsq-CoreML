import Foundation
import CoreML

extension MLModel {
    
    struct Output4d {
        let solution: NdArray3d
        let rank: NdArray2d
        let singularValues: NdArray3d
    }
    
    struct Output3d {
        let solution: NdArray2d
        let rank: NdArray1d
        let singularValues: NdArray2d
    }
    
    struct Output2d {
        let solution: NdArray1d
        let rank: Float32
        let singularValues: NdArray1d
    }
    
    func forward(_ a: [[Float64]], _ b: [Float64]) throws -> Output2d {
        let a = a.map { x in x.map { Float32($0) } }
        let b = b.map { Float32($0) }
        return try self.forward(a, b)
    }
    
    func forward(_ a: [[Float32]], _ b: [Float32]) throws -> Output2d {
        let result = try self.forward([a], [b])
        return Output2d(
            solution: result.solution.first(),
            rank: result.rank.first(),
            singularValues: result.singularValues.first()
        )
    }
    
    func forward(_ a: [[[Float32]]], _ b: [[Float32]]) throws -> Output3d {
        let result = try self.forward([a], [b])
        return Output3d(
            solution: result.solution.first(),
            rank: result.rank.first(),
            singularValues: result.singularValues.first()
        )
    }
    
    func forward(_ a: [[[[Float32]]]], _ b: [[[Float32]]]) throws -> Output4d {
        let aArray = NdArray4d(value: a)
        let bArray = NdArray3d(value: b)
        
        return try self.forward(aArray, bArray)
    }
    
    func forward(_ a: NdArray4d, _ b: NdArray3d) throws -> Output4d {
        
        let aTensor = try NdArrayUtil.toMLArray(array: a)
        let bTensor = try NdArrayUtil.toMLArray(array: b)
        
        let input = Input(a: aTensor, b: bTensor)
        let outputs = try self.prediction(from: input)
                
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
        
        return Output4d(
            solution: solutionArray,
            rank: rankArray,
            singularValues: singularValuesArray
        )
    }
}

private final class Input: MLFeatureProvider {
    
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

// MARK: - Debug

protocol DebugOutput: CustomDebugStringConvertible {
    var info: (solution: NdArray, rank: NdArray, singularValues: NdArray) { get }
}

extension MLModel.Output4d: DebugOutput {
    var info: (solution: any NdArray, rank: any NdArray, singularValues: any NdArray) {
        (solution, rank, singularValues)
    }
}

extension MLModel.Output3d: DebugOutput {
    var info: (solution: any NdArray, rank: any NdArray, singularValues: any NdArray) {
        (solution, rank, singularValues)
    }
}

extension MLModel.Output2d: DebugOutput {
    var info: (solution: any NdArray, rank: any NdArray, singularValues: any NdArray) {
        (solution, rank, singularValues)
    }
}

extension DebugOutput {
    public var debugDescription: String {
        let info = self.info
        return "solution=\(info.solution),\n" +
            "rank=\(info.rank),\n" +
            "singular_values=\(info.singularValues)"
    }
}
