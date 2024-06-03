import Foundation
import CoreML
import Accelerate

func lstsq(inputs: [MLMultiArray], outputs: [MLMultiArray]) {
    inputs.forEach {
        assert($0.dataType == .float32)
    }
    outputs.forEach {
        assert($0.dataType == .float32)
    }
    let shapeA = inputs[0].shape
    let shapeB = inputs[1].shape
    
    let a = inputs[0].toPtr(type: Float32.self)
    let b = inputs[1].toPtr(type: Float32.self)

    let M = shapeA[shapeA.count - 1].intValue
    let N = shapeA[shapeA.count - 2].intValue

    var m = Int32(M)
    var n = Int32(N)
    
    let x = outputs[0].toPtr(type: Float32.self, reset: true)
    memcpy(x, b, MemoryLayout<Float32>.stride * shapeB.tensorSize())
    
    var nrhs: Int32 = 1
    var lda = m
    var ldb = n
    
    var info: Int32 = 0
    var lwork: Int32 = 0
    var rank: Int32 = 0
    
    var rcond: Float32 = -1.0
    var wkopt: Float32 = 0
    
    var iwork = [Int32](repeating: 0, count: 11 * M)
    
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
    
    lwork = Int32(wkopt)
    var work = [Float32](repeating: 0, count: Int(lwork))

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
    
    var rankFloat = Float32(rank)
    memcpy(outputs[2].dataPointer, &rankFloat, MemoryLayout<Float32>.stride)
}
