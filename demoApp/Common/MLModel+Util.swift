import Foundation
import CoreML

extension MLModel {
    
    typealias Output = (
        solution: NdArray3d,
        rank: NdArray2d,
        singularValues: NdArray3d
    )
    
    func forward(_ a: [[Float32]], _ b: [Float32]) throws -> Output {
        return try self.forward([a], [b])
    }
    
    func forward(_ a: [[[Float32]]], _ b: [[Float32]]) throws -> Output {
        return try self.forward([a], [b])
    }
    
    func forward(_ a: [[[[Float32]]]], _ b: [[[Float32]]]) throws -> Output {
        let aArray = NdArray4d(value: a)
        let bArray = NdArray3d(value: b)
        
        return try self.forward(aArray, bArray)
    }
    
    func forward(_ a: NdArray4d, _ b: NdArray3d) throws -> Output {
        
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
        
        return (solutionArray, rankArray, singularValuesArray)
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
