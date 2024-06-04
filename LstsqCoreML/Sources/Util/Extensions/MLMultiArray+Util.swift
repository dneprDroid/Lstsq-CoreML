import Foundation
import CoreML

extension MLMultiArray {
    
    public func toFlattenArray<T: Numeric>(for type: T.Type  = T.self) -> [T] {
        let ptr = self.dataPointer.assumingMemoryBound(to: T.self)
        let array = [T](UnsafeBufferPointer(start: ptr, count: self.count))
        return array
    }
    
    public func toPtr<T>(type: T.Type, reset: Bool = false) -> UnsafeMutablePointer<T> {
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
