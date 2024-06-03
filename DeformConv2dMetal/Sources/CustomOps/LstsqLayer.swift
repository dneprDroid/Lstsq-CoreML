import Foundation
import CoreML

public enum LstsqLayerError: Error {
    case invalidDriver
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
        return [
            [shapeA[shapeA.count - 2]],
            [1],
            [1],
            [shapeA[shapeA.count - 1]],
        ]
    }
    
    public func evaluate(inputs: [MLMultiArray], outputs: [MLMultiArray]) throws {
        lstsq(inputs: inputs, outputs: outputs)
    }
}
