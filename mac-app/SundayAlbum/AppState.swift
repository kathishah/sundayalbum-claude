import Foundation
import Observation

enum AppScreen: Equatable {
    case library
    case stepDetail(jobID: UUID)
}

@Observable
final class AppState {
    var jobs: [ProcessingJob] = []
    var expandedJobID: UUID?      // single-click expand in library
    var currentScreen: AppScreen = .library

    var selectedJob: ProcessingJob? {
        if case .stepDetail(let id) = currentScreen {
            return jobs.first { $0.id == id }
        }
        return nil
    }

    func jobFor(id: UUID) -> ProcessingJob? {
        jobs.first { $0.id == id }
    }

    /// Call from views, wrapped in withAnimation if desired
    func navigate(to screen: AppScreen) {
        currentScreen = screen
        expandedJobID = nil
    }

    func navigateBack() {
        currentScreen = .library
    }

    /// Enqueues new files, skipping any whose inputURL is already in the library.
    func addFiles(_ urls: [URL]) {
        let existingURLs = Set(jobs.compactMap(\.inputURL))
        let newJobs = urls
            .filter { !existingURLs.contains($0) }
            .map { ProcessingJob(inputName: $0.lastPathComponent, inputURL: $0) }
        jobs.append(contentsOf: newJobs)
    }
}
