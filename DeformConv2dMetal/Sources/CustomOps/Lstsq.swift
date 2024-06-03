import Foundation
import CoreML
import Accelerate

extension MLMultiArray {
    func toPtr<T>(type: T.Type, reset: Bool = false) -> UnsafeMutablePointer<T> {
        let cap = self.shape.tensorSize()
        let pointer = self.dataPointer.bindMemory(
            to: T.self,
            capacity: cap
        )
        if reset {
            memset(pointer, 0, cap)
        }
        return pointer
    }
}

func lstsq(inputs: [MLMultiArray], outputs: [MLMultiArray]) {
    inputs.forEach {
        assert($0.dataType == .float32)
    }
    outputs.forEach {
        assert($0.dataType == .float32)
    }
    let shapeA = inputs[0].shape
    let shapeB = inputs[1].shape
    
    
    let a = inputs[0].toPtr(type: Float.self)
    let b = inputs[1].toPtr(type: Float32.self)

    let M = shapeA[shapeA.count - 1].intValue
    let N = shapeA[shapeA.count - 2].intValue

    var m = __CLPK_integer(M)
    var n = __CLPK_integer(N)
    
    let x = outputs[0].toPtr(type: Float32.self, reset: true)
    memcpy(x, b, MemoryLayout<Float>.stride * shapeB.tensorSize())
    
    var nrhs: __CLPK_integer = 1
    var lda = m
    var ldb = n
    
    var info: __CLPK_integer = 0
    var lwork: __CLPK_integer = 0
    var rank: __CLPK_integer = 0
    
    var rcond: Float = -1.0
    var wkopt: Float = 0
    
    var iwork = [__CLPK_integer](repeating: 0, count: 11 * M)
    
    let s = outputs[3].toPtr(type: Float32.self, reset: true)
    
    lwork = -1
    
    sgelsd_(
        &m,
        &n,
        &nrhs,
        a,
        &lda,
        x,
        &ldb,
        s,
        &rcond,
        &rank,
        &wkopt,
        &lwork,
        &iwork,
        &info
    )
    
    lwork = __CLPK_integer(wkopt)
    var work = [Float](repeating: 0, count: Int(lwork))

    sgelsd_(
        &m,
        &n,
        &nrhs,
        a,
        &lda,
        x,
        &ldb,
        s,
        &rcond,
        &rank,
        &work,
        &lwork,
        &iwork,
        &info
    )
    
    var rankFloat = Float(rank)
    memcpy(outputs[2].dataPointer, &rankFloat, MemoryLayout<Float>.stride)
}
