import Foundation
import CoreML

extension MLMultiArray {
    
    func toNdArray2d() -> NdArray2d {
        let shape = self.shape.map { $0.intValue }
        assert(shape.count == 2)

        let outValue = NdArray2d.TensorType(
            repeating: NdArray1d.TensorType(repeating: 0, count: shape[1]),
            count: shape[0]
        )
        let out = NdArray2d(value: outValue)
        out.forEach { ndIndex, _ in
            let index = ndIndex.map { NSNumber(value: $0) }
            out[ndIndex] = self[index].floatValue
        }
        return out
    }
    
    func toNdArray3d() -> NdArray3d {
        let shape = self.shape.map { $0.intValue }
        assert(shape.count == 3)
        
        let outValue = NdArray3d.TensorType(
            repeating: NdArray2d.TensorType(
                repeating: NdArray1d.TensorType(
                    repeating: 0,
                    count: shape[2]
                ),
                count: shape[1]
            ),
            count: shape[0]
        )
        let out = NdArray3d(value: outValue)
        out.forEach { ndIndex, _ in
            let index = ndIndex.map { NSNumber(value: $0) }
            out[ndIndex] = self[index].floatValue
        }
        return out
    }
}
