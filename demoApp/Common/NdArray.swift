import Foundation
import CoreML

//typealias NdArray1d = [Float32]
//typealias NdArray2d = [[Float32]]
//typealias NdArray3d = [[[Float32]]]
//typealias NdArray4d = [[[[Float32]]]]

enum NdArrayError: Error {
    case resNotFound
}

protocol NdArray: Decodable {
    var shape: [Int] { get }
    
    subscript(ndIndex: [Int]) -> Float32 { get set }
    
    func forEach(_ body: (_ ndIndex: [Int], _ value: Float32) -> Void)
}

final class NdArray4d: NdArray {
    typealias TensorType = [[[[Float32]]]]
    
    var value: TensorType
    
    init(value: TensorType) {
        self.value = value
    }
    
    init(from decoder: any Decoder) throws {
        let container = try decoder.singleValueContainer()
        value = try container.decode(TensorType.self)
    }
    
    var shape: [Int] {
        [
            value.count,
            value.first?.count ?? 0,
            value.first?.first?.count ?? 0,
            value.first?.first?.first?.count ?? 0,
        ]
    }
    
    subscript(ndIndex: [Int]) -> Float32 {
        get {
            validateIndex(ndIndex)
            
            return value[ndIndex[0]][ndIndex[1]][ndIndex[2]][ndIndex[3]]
        }
        set {
            validateIndex(ndIndex)

            value[ndIndex[0]][ndIndex[1]][ndIndex[2]][ndIndex[3]] = newValue
        }
    }
    
    func forEach(_ body: (_ ndIndex: [Int], _ value: Float32) -> Void) {
        let shape = self.shape
        
        for n in 0..<shape[0] {
            for c in 0..<shape[1] {
                for x in 0..<shape[2] {
                    for y in 0..<shape[3] {
                        let ndIndex = [n, c, x, y]
                        let value = value[n][c][x][y]
                        
                        body(ndIndex, value)
                    }
                }
            }
        }
    }
    
    private func validateIndex(_ ndIndex: [Int]) {
        guard ndIndex.count == 4 else { fatalError("Invalid index: \(ndIndex)") }
    }
}


final class NdArray3d: NdArray {
    typealias TensorType = [[[Float32]]]
    
    var value: TensorType
    
    init(value: TensorType) {
        self.value = value
    }
    
    init(from decoder: any Decoder) throws {
        let container = try decoder.singleValueContainer()
        value = try container.decode(TensorType.self)
    }
    
    var shape: [Int] {
        [
            value.count,
            value.first?.count ?? 0,
            value.first?.first?.count ?? 0,
        ]
    }
    
    subscript(ndIndex: [Int]) -> Float32 {
        get {
            validateIndex(ndIndex)
            
            return value[ndIndex[0]][ndIndex[1]][ndIndex[2]]
        }
        set {
            validateIndex(ndIndex)

            value[ndIndex[0]][ndIndex[1]][ndIndex[2]] = newValue
        }
    }
    
    func forEach(_ body: (_ ndIndex: [Int], _ value: Float32) -> Void) {
        let shape = self.shape
        
        for c in 0..<shape[0] {
            for x in 0..<shape[1] {
                for y in 0..<shape[2] {
                    let ndIndex = [c, x, y]
                    let value = value[c][x][y]
                    
                    body(ndIndex, value)
                }
            }
        }
    }
    
    private func validateIndex(_ ndIndex: [Int]) {
        guard ndIndex.count == 3 else { fatalError("Invalid index: \(ndIndex)") }
    }
}

final class NdArray2d: NdArray {
    typealias TensorType = [[Float32]]
    
    var value: TensorType
    
    init(value: TensorType) {
        self.value = value
    }
    
    init(from decoder: any Decoder) throws {
        let container = try decoder.singleValueContainer()
        value = try container.decode(TensorType.self)
    }
    
    var shape: [Int] {
        [
            value.count,
            value.first?.count ?? 0,
        ]
    }
    
    subscript(ndIndex: [Int]) -> Float32 {
        get {
            validateIndex(ndIndex)
            
            return value[ndIndex[0]][ndIndex[1]]
        }
        set {
            validateIndex(ndIndex)

            value[ndIndex[0]][ndIndex[1]] = newValue
        }
    }
    
    func forEach(_ body: (_ ndIndex: [Int], _ value: Float32) -> Void) {
        let shape = self.shape
        
        for x in 0..<shape[0] {
            for y in 0..<shape[1] {
                let ndIndex = [x, y]
                let value = value[x][y]
                
                body(ndIndex, value)
            }
        }
    }
    
    private func validateIndex(_ ndIndex: [Int]) {
        guard ndIndex.count == 2 else { fatalError("Invalid index: \(ndIndex)") }
    }
}

final class NdArray1d: NdArray {
    typealias TensorType = [Float32]
    
    var value: TensorType
    
    init(value: TensorType) {
        self.value = value
    }
    
    init(from decoder: any Decoder) throws {
        let container = try decoder.singleValueContainer()
        value = try container.decode(TensorType.self)
    }
    
    var shape: [Int] {
        [ value.count ]
    }
    
    subscript(ndIndex: [Int]) -> Float32 {
        get {
            validateIndex(ndIndex)
            
            return value[ndIndex[0]]
        }
        set {
            validateIndex(ndIndex)

            value[ndIndex[0]] = newValue
        }
    }
    
    func forEach(_ body: (_ ndIndex: [Int], _ value: Float32) -> Void) {
        let shape = self.shape
        
        for i in 0..<shape[0] {
            let ndIndex = [i]
            let value = value[i]
            
            body(ndIndex, value)
        }
    }
    
    private func validateIndex(_ ndIndex: [Int]) {
        guard ndIndex.count == 1 else { fatalError("Invalid index: \(ndIndex)") }
    }
}
