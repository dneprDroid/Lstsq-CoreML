import Foundation
import CoreML
import Accelerate

private func toPtr<T>(_ array: MLMultiArray, type: T.Type, reset: Bool) -> (UnsafeMutablePointer<T>, Int) {
    let cap = array.shape.tensorSize()
    let pointer = array.dataPointer.bindMemory(
        to: T.self,
        capacity: cap
    )
    if reset {
        memset(pointer, 0, cap)
    }
    return (pointer, cap)
}

func toArray<T>(_ ptr: UnsafeMutablePointer<T>, _ count: Int) -> [T] {
    return (0..<count).map { ptr[$0] }
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
    
    let (a, a_count) = toPtr(inputs[0], type: Float.self, reset: false)
    let (b, b_count) = toPtr(inputs[1], type: Float.self, reset: false)
    
    print("---- a: ", toArray(a, a_count))
    print("---- b: ", toArray(b, b_count))
    
    let rightHandSideCount = 1
    let leadingDimensionB = max(M, N)
    let xCount = Int(leadingDimensionB * rightHandSideCount)
    
    
    var (x, x_count) = toPtr(outputs[0], type: Float.self, reset: true) // [Float](repeating: 0, count: xCount)
    assert(x_count == xCount)
    memcpy(x, b, MemoryLayout<Float>.stride * shapeB.tensorSize())
    
//    int m = M, n = N, nrhs = NRHS, lda = LDA, ldb = LDB, info, lwork, rank;
    var nrhs: __CLPK_integer = 1
    var lda = m
    var ldb = n
    
    var info: __CLPK_integer = 0
    var lwork: __CLPK_integer = 0
//    var rank: __CLPK_integer = 0
    let (rank, randk_count) = toPtr(outputs[2], type: __CLPK_integer.self, reset: true)
    assert(randk_count == 1)
    
    var rcond: Float = -1.0
    var wkopt: Float = 0
    
    var iwork = [__CLPK_integer](repeating: 0, count: 11*M)
    
    let (s, s_count) = toPtr(outputs[3], type: Float.self, reset: true)// [Float](repeating: 0, count: Int(m));
    assert(s_count == m)
//    var a = [Float](repeating: 0, count: lda * n);
//    assert(lda * n == a.count)
    
    lwork = -1
    
    sgelsd_(&m, &n, &nrhs, a, &lda, x, &ldb, s, &rcond, rank, &wkopt, &lwork,
                    &iwork, &info )
    
    lwork = __CLPK_integer(wkopt)
    var work = [Float](repeating: 0, count: Int(wkopt))

    print("lwork: ", lwork, work.count)
    print("rank: ", rank[0], outputs[2].shape.tensorSize())
    sgelsd_(&m, &n, &nrhs, a, &lda, x, &ldb, s, &rcond, rank, &work, &lwork,
            &iwork, &info )
    
    print("solution: ", x[0], x[1], x[2], x[3], x[4])
    print("singular_values: ", s[0], s[1], s[2])
    print("rank: ", rank[0])
}
