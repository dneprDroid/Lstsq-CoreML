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

func readInput<T>(_ array: MLMultiArray, type: T.Type) -> [T] {
    let (ptr, count) = toPtr(array, type: type, reset: false)
    return toArray(ptr, count)
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
    
    let A = readInput(inputs[0], type: Float32.self)
    let B = readInput(inputs[1], type: Float32.self)

    let M = shapeA[shapeA.count - 1].intValue
    let N = shapeA[shapeA.count - 2].intValue
    print("M, N: ", (M, N))
    print("A, B: ", (A, B))

    var m = __CLPK_integer(M)
    var n = __CLPK_integer(N)
    
    var a = A
    let rightHandSideCount = 1
    let leadingDimensionB = max(M, N)
    let xCount = Int(leadingDimensionB * rightHandSideCount)
    
    
    var x = [Float](repeating: 0, count: xCount)
    memcpy(&x, B, MemoryLayout<Float>.stride * B.count)
    
//    int m = M, n = N, nrhs = NRHS, lda = LDA, ldb = LDB, info, lwork, rank;
    var nrhs: __CLPK_integer = 1
    var lda = m
    var ldb = n
    
    var info: __CLPK_integer = 0
    var lwork: __CLPK_integer = 0
    var rank: __CLPK_integer = 0
    
    var rcond: Float = -1.0
    var wkopt: Float = 0
    
    var iwork = [__CLPK_integer](repeating: 0, count: 11*M)
    
    var s = [Float](repeating: 0, count: Int(m));
//    var a = [Float](repeating: 0, count: lda * n);
    assert(lda * n == a.count)
    
    lwork = -1
    
    sgelsd_(&m, &n, &nrhs, &a, &lda, &x, &ldb, &s, &rcond, &rank, &wkopt, &lwork,
                    &iwork, &info )
    
    lwork = __CLPK_integer(wkopt)
    var work = [Float](repeating: 0, count: Int(lwork))

    print("lwork: ", lwork)
    sgelsd_(&m, &n, &nrhs, &a, &lda, &x, &ldb, &s, &rcond, &rank, &work, &lwork,
            &iwork, &info )
    
    print("solution: ", x, outputs[0].shape.tensorSize())
    print("singular_values: ", s, outputs[3].shape.tensorSize())
    print("rank: ", rank)
    
    var rankFloat = Float(rank)
    memcpy(outputs[0].dataPointer, &x, x.count * MemoryLayout<Float>.stride)
    memcpy(outputs[2].dataPointer, &rankFloat, MemoryLayout<Float>.stride)
    memcpy(outputs[3].dataPointer, &s, s.count * MemoryLayout<Float>.stride)
}
