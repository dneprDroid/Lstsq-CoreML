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
    
    let aBatched = inputs[0].toPtr(type: Float32.self)
    let bBatched = inputs[1].toPtr(type: Float32.self)
    let xBatched = outputs[0].toPtr(type: Float32.self, reset: true)

    let rankPtrBatched = outputs[2].dataPointer
    let sBatched = outputs[3].toPtr(type: Float32.self, reset: true)

    
    let M = shapeA[shapeA.count - 1].intValue
    let N = shapeA[shapeA.count - 2].intValue

    var m = Int32(M)
    var n = Int32(N)
    
    let batchSize = shapeA.tensorSize() / (M * N)
    
//    assert(M < N)
    
    for bIndex in 0..<batchSize {
        
        let a = aBatched.advanced(by: bIndex * M * N)
        let b = bBatched.advanced(by: bIndex * M)
        
        var x = [Float32](repeating: 0, count: max(M, N))
        memcpy(&x, b, MemoryLayout<Float32>.stride * M)
        
        let s = sBatched.advanced(by: bIndex * M)
        
        let rankPtr = rankPtrBatched.advanced(by: bIndex * MemoryLayout<Float32>.stride)

        var nrhs: Int32 = 1
        var lda = m
        var ldb = n
        
        var info: Int32 = 0
        var lwork: Int32 = 0
        var rank: Int32 = 0
        
        var rcond: Float32 = -1.0
        var wkopt: Float32 = 0
        
        var iwork = [Int32](repeating: 0, count: 11 * M)
        
        
        lwork = -1
        
        sgelsd_(
            &m,
            &n,
            &nrhs,
            a,
            &lda,
            &x,
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
            &x,
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
        memcpy(rankPtr, &rankFloat, MemoryLayout<Float32>.stride)
        
        memcpy(xBatched.advanced(by: bIndex * x.count), &x, x.count * MemoryLayout<Float32>.stride)
    }
}
