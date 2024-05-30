import Foundation
import CoreML

public enum LstsqLayerError: Error {
    case invalidDriver
}

@objc(dneprDroid_lstsq)
public final class LstsqLayer: NSObject, MLCustomLayer {
    
    private let driver: String
    
    init(parameters: [String : Any]) throws {
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
        let aShape = inputShapes[0]
        return [
            [aShape[aShape.count - 1]],
            [0],
            [1],
            [aShape[aShape.count - 2]],
        ]
    }
    
    func evaluate(inputs: [MLMultiArray], outputs: [MLMultiArray]) throws {
        <#code#>
    }
}
