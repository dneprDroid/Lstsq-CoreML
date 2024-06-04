import Foundation

extension Array where Element == Int {
    func tensorSize() -> Int {
        return self.reduce(1, { $0 * $1 })
    }
}

extension Array where Element == NSNumber {
    func tensorSize() -> Int {
        return self.reduce(1, { $0 * $1.intValue })
    }
}
