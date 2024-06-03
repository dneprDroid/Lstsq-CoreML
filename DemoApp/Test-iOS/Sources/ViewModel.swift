import Foundation
import SwiftUI
import DeformConv2dMetal

final class ViewModel: ObservableObject {
    
    @Published var state: State = .initial
    
    private var worker = MLModelTestWorker()

    func test() async throws {
        worker.onUpdateState = updateState
        try await worker.test()
    }
    
    @MainActor private func updateState(_ newState: State) {
        self.state = newState
    }
}
