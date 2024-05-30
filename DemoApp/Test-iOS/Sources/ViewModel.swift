import Foundation
import SwiftUI
import DeformConv2dMetal

final class ViewModel: ObservableObject {
    
    @Published var state: State = .initial
    
    private var worker = MLModelTestWorker()

    func test() async throws {
//        try await Task.sleep(nanoseconds: 700_000_000)
        DispatchQueue.global().asyncAfter(deadline: .now() + 4) {
//            gels_test()
            let aValues: [Float] = [1, 6, 11,
                                    2, 7, 12,
                                    3, 8, 13,
                                    4, 9, 14,
                                    5, 10, 15]
            
            let dimension = (m: 3, n: 5)
            
            /// The _b_ in _Ax = b_.
            let bValues: [Float] = [355, 930, 1505]
            calculateSolution222(a: aValues, b: bValues, m: dimension.m, n: dimension.n)
        }
//        worker.onUpdateState = updateState
//        
//        try await worker.test()
    }
    
    @MainActor private func updateState(_ newState: State) {
        self.state = newState
    }
}
