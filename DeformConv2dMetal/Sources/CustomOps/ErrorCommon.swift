import Foundation
import Metal
import Accelerate

public enum ErrorCommon: Error {
    case metalNotSupported
    case shaderLibNotFound
    case shaderNotFound
    case cpuNotImplemented
    case encoderInvalid
    case invalidLayerParams
    case missingWeights
    case missingBundle
    case textureAllocateFailed
    case textureDataConversionNotSupported(
        bytesPerComponent: Int,
        pixelFormat: MTLPixelFormat
    )
    case dataConversionUnsupported(
        srcBytesPerComponent: Int,
        dstBytesPerComponent: Int
    )
    case pixelFormatNotSupported(MTLPixelFormat)
}

public func calculateSolution222(
    a A: [Float],
    b B: [Float],
    m M: Int,
    n N: Int
) {
    var m = __CLPK_integer(M)
    var n = __CLPK_integer(N)
    print("M, N: ", (M, N))
    print("A, B: ", (A, B))
    
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
    
    print("solution: ", x)
    print("singular_values: ", s)
    print("rank: ", rank)
}
