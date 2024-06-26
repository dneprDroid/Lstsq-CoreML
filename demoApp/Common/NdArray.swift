import Foundation
import CoreML

enum NdArrayError: Error {
    case resNotFound
}

protocol NdArray: Decodable, CustomDebugStringConvertible {
    var shape: [Int] { get }
    
    subscript(ndIndex: [Int]) -> Float32 { get set }
    
    func forEach(_ body: (_ ndIndex: [Int], _ value: Float32) -> Void)
}

final class NdArray4d: NdArray {
    typealias TensorType = [[[[Float32]]]]
    
    var value: TensorType
    
    var debugDescription: String { value.debugDescription }
    
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
    
    func first() -> NdArray3d {
        return .init(value: value[0])
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
    
    var debugDescription: String { value.debugDescription }
    
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
    
    func first() -> NdArray2d {
        return .init(value: value[0])
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
    
    var debugDescription: String { value.debugDescription }
    
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
    
    func first() -> NdArray1d {
        return .init(value: value[0])
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
    
    var debugDescription: String { value.debugDescription }
    
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
    
    func first() -> Float32 {
        return value[0]
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

extension Float32: NdArray {
    var shape: [Int] { [1] }
        
    subscript(ndIndex: [Int]) -> Float32 {
        get {
            validateIndex(ndIndex)
            return self
        }
        set {
            validateIndex(ndIndex)
            self = newValue
        }
    }
    
    func forEach(_ body: (_ ndIndex: [Int], _ value: Float32) -> Void) {
        body([0], self)
    }
    
    private func validateIndex(_ ndIndex: [Int]) {
        guard ndIndex.count == 0 || (ndIndex.count == 1 && ndIndex[0] == 0) else {
            fatalError("Invalid index: \(ndIndex)")
        }
    }
}
