import Foundation
import CoreML

typealias NdArray1d = [Float32]
typealias NdArray2d = [[Float32]]
typealias NdArray3d = [[[Float32]]]
typealias NdArray4d = [[[[Float32]]]]

enum NdArrayError: Error {
    case resNotFound
}

protocol NdArray: Decodable {
    var shape: [Int] { get }
        
    func forEach(_ body: (_ ndIndex: [Int], _ value: Float32) -> Void)
}

extension NdArray {
    subscript(ndIndex: [Int]) -> Float32 {
        get {
            if let array2d = self as? NdArray2d {
                return array2d.getElement2d(ndIndex)
            }
            fatalError()
        }
        set {
            if var array2d = self as? NdArray2d {
                return array2d.setElement2d(ndIndex, newValue)
            }
            fatalError()
        }
    }
}

extension NdArray2d {
    
    func getElement2d(_ ndIndex: [Int]) -> Float32 {
        validateIndex(ndIndex)
        return self[ndIndex[0]][ndIndex[1]]
    }
    
    mutating func setElement2d(_ ndIndex: [Int], _ newValue: Float32) {
        validateIndex(ndIndex)
        self[ndIndex[0]][ndIndex[1]] = newValue
    }
    
    private func validateIndex(_ ndIndex: [Int]) {
        guard ndIndex.count == 2 else { fatalError("Invalid index: \(ndIndex)") }
    }
}

extension NdArray2d: NdArray where Element == NdArray1d {
    
    private var shape4d: [Int] {
        [
            self.count,
            self.first?.count ?? 0
        ]
    }
    
    
    
    func forEach(_ body: (_ ndIndex: [Int], _ value: Float32) -> Void) {
        let shape = self.shape
        
        for x in 0..<shape[0] {
            for y in 0..<shape[1] {
                let ndIndex = [x, y]
                let value = self[x][y]
                
                body(ndIndex, value)
            }
        }
    }
}

extension NdArray4d: NdArray where Element == NdArray3d {
    
    var shape: [Int] {
        [
            self.count,
            self.first?.count ?? 0,
            self.first?.first?.count ?? 0,
            self.first?.first?.first?.count ?? 0,
        ]
    }
    
    subscript(ndIndex: [Int]) -> Float32 {
        get {
            validateIndex(ndIndex)
            
            return self[ndIndex[0]][ndIndex[1]][ndIndex[2]][ndIndex[3]]
        }
        set {
            validateIndex(ndIndex)

            self[ndIndex[0]][ndIndex[1]][ndIndex[2]][ndIndex[3]] = newValue
        }
    }
    
    func forEach(_ body: (_ ndIndex: [Int], _ value: Float32) -> Void) {
        let shape = self.shape
        
        for n in 0..<shape[0] {
            for c in 0..<shape[1] {
                for x in 0..<shape[2] {
                    for y in 0..<shape[3] {
                        let ndIndex = [n, c, x, y]
                        let value = self[n][c][x][y]
                        
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
