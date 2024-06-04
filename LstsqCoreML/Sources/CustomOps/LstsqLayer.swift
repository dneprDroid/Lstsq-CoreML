import Foundation
import CoreML

public enum LstsqLayerError: Error {
    case invalidDriver
    case invalidDataType
}

@objc(dneprDroid_lstsq)
public final class LstsqLayer: NSObject, MLCustomLayer {
    
    private let driver: String
    
    public init(parameters: [String : Any]) throws {
        guard
            let driver = parameters["driver"] as? String,
            driver == "gelsd"
        else {
            throw LstsqLayerError.invalidDriver
        }
        self.driver = driver
        super.init()
    }
    
    public func setWeightData(_ weights: [Data]) throws {}
    
    public func outputShapes(forInputShapes inputShapes: [[NSNumber]]) throws -> [[NSNumber]] {
        let shapeA = inputShapes[0]
        let batches = shapeA[0..<shapeA.count - 2]
        let rankShape = batches.count == 0 ? [1] : [NSNumber](batches)
        
        return [
            batches + [shapeA[shapeA.count - 2]],
            [1],
            rankShape,
            batches + [shapeA[shapeA.count - 1]],
        ]
    }
    
    public func evaluate(inputs: [MLMultiArray], outputs: [MLMultiArray]) throws {
        try lstsq(inputs: inputs, outputs: outputs)
    }
}
