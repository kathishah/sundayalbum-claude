import Foundation
import Observation

@Observable
final class ExtractedPhoto: Identifiable {
    let id: UUID
    let imageURL: URL          // The processed output (mock: a test HEIC)
    let jobInputURL: URL?      // The original input (for before/after)
    let jobInputName: String
    let jobID: UUID

    init(
        id: UUID = UUID(),
        imageURL: URL,
        jobInputURL: URL? = nil,
        jobInputName: String,
        jobID: UUID
    ) {
        self.id = id
        self.imageURL = imageURL
        self.jobInputURL = jobInputURL
        self.jobInputName = jobInputName
        self.jobID = jobID
    }
}
