import Foundation
import Observation

@Observable
final class ExtractedPhoto: Identifiable {
    let id: UUID
    let imageURL: URL          // The processed output
    let jobInputURL: URL?      // The original input (for before/after)
    let jobInputName: String
    let jobID: UUID

    /// User-set rotation override for the orientation step (0 / 90 / 180 / 270).
    /// nil means use whatever the AI detected. Passed as visual rotation in the UI;
    /// future: wired into --scene-desc / custom rotation CLI flag.
    var rotationOverride: Int?

    /// User-written scene description shown in the orientation step.
    /// When set, passed as --scene-desc to the CLI when re-running from orientation.
    var sceneDescription: String?

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
