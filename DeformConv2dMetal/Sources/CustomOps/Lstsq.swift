import Foundation
import CoreML
import Accelerate

private func toPtr<T>(_ array: MLMultiArray, type: T.Type) -> UnsafeMutablePointer<T> {
    return array.dataPointer.bindMemory(
        to: T.self,
        capacity: array.shape.tensorSize()
    )
}

func lstsq(inputs: [MLMultiArray], outputs: [MLMultiArray]) {
    let shapeA = inputs[0].shape
    let shapeB = inputs[1].shape

//    let A = [Float](UnsafeBufferPointer(
//        start: inputs[0].dataPointer.bindMemory(
//            to: Float.self,
//            capacity: shapeA.tensorSize()
//        ),
//        count: shapeA.tensorSize())
//    )

    let M = shapeA[shapeA.count - 2].intValue
    let N = shapeA[shapeA.count - 1].intValue
    
    var m = __CLPK_integer(M)
    var n = __CLPK_integer(N)
    
    let a = toPtr(inputs[0], type: Float.self)
    let b = toPtr(inputs[1], type: Float.self)
    
    let rightHandSideCount = 1
    let leadingDimensionB = max(M, N)
//    let xCount = Int(leadingDimensionB * rightHandSideCount)
    
    
    var x = toPtr(outputs[0], type: Float.self) // [Float](repeating: 0, count: xCount)
    memcpy(&x, b, MemoryLayout<Float>.stride * shapeB.tensorSize())
    
//    int m = M, n = N, nrhs = NRHS, lda = LDA, ldb = LDB, info, lwork, rank;
    var nrhs: __CLPK_integer = 1
    var lda = m
    var ldb = n
    
    var info: __CLPK_integer = 0
    var lwork: __CLPK_integer = 0
//    var rank: __CLPK_integer = 0
    let rank = toPtr(outputs[2], type: __CLPK_integer.self)
    
    var rcond: Float = -1.0
    var wkopt: Float = 0
    
    var iwork = [__CLPK_integer](repeating: 0, count: 11*M)
    
    let s = toPtr(outputs[3], type: Float.self)// [Float](repeating: 0, count: Int(m));
//    var a = [Float](repeating: 0, count: lda * n);
//    assert(lda * n == a.count)
    
    lwork = -1
    
    sgelsd_(&m, &n, &nrhs, a, &lda, x, &ldb, s, &rcond, rank, &wkopt, &lwork,
                    &iwork, &info )
    
    lwork = __CLPK_integer(wkopt)
    var work = [Float](repeating: 0, count: Int(lwork))

    sgelsd_(&m, &n, &nrhs, a, &lda, x, &ldb, s, &rcond, rank, &work, &lwork,
            &iwork, &info )
    
    print("solution: ", x)
    print("singular_values: ", s)
    print("rank: ", rank)
}
