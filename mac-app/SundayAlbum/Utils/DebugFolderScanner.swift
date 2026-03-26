import Foundation

/// Scans the project's `debug/` folder at app launch and reconstructs one completed
/// `ProcessingJob` per subfolder that contains final photo outputs.
///
/// Debug folder layout expected:
///   debug/<stem>/01_loaded.jpg           — before image
///   debug/<stem>/15_photo_NN_final.jpg   — final output per photo (preferred)
///   debug/<stem>/14_photo_NN_enhanced.jpg — fallback final output
struct DebugFolderScanner {

    @MainActor private static var debugRoot: URL {
        RuntimeManager.shared.cliWorkingDirectory.appendingPathComponent("debug")
    }

    @MainActor private static var testImagesRoot: URL {
        RuntimeManager.shared.cliWorkingDirectory.appendingPathComponent("test-images")
    }

    // MARK: - Public

    /// Returns one completed `ProcessingJob` for each debug subfolder that has
    /// at least one final photo output. Folders with no outputs are skipped.
    @MainActor static func loadJobs() -> [ProcessingJob] {
        guard let entries = try? FileManager.default.contentsOfDirectory(
            at: debugRoot,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        ) else { return [] }

        return entries
            .filter { isDirectory($0) }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
            .compactMap { jobFrom(folder: $0) }
    }

    // MARK: - Private

    @MainActor private static func jobFrom(folder: URL) -> ProcessingJob? {
        let stem = folder.lastPathComponent
        let photoURLs = finalPhotoURLs(in: folder)
        guard !photoURLs.isEmpty else { return nil }

        // Before image — 01_loaded.jpg in the same debug folder
        let loadedURL = folder.appendingPathComponent("01_loaded.jpg")
        let beforeURL: URL? = FileManager.default.fileExists(atPath: loadedURL.path) ? loadedURL : nil

        // Try to locate the original input file in test-images/
        let foundInput = findInputFile(stem: stem)
        let inputURL = foundInput ?? beforeURL
        let inputName = foundInput?.lastPathComponent ?? stem

        let job = ProcessingJob(
            inputName: inputName,
            inputURL: inputURL,
            state: .complete,
            currentStep: .done,
            stepStatus: .complete
        )

        job.extractedPhotos = photoURLs.map { url in
            ExtractedPhoto(
                imageURL: url,
                jobInputURL: beforeURL,
                jobInputName: inputName,
                jobID: job.id
            )
        }

        return job
    }

    /// Returns the highest-step output URL for each photo index found in the folder.
    /// Prefers step 15 (final) over step 14 (enhanced) over step 13 (restored).
    private static func finalPhotoURLs(in folder: URL) -> [URL] {
        guard let files = try? FileManager.default.contentsOfDirectory(
            at: folder,
            includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles]
        ) else { return [] }

        // Map: photoIndex → (highestStep, URL)
        var best: [Int: (step: Int, url: URL)] = [:]

        for file in files where file.pathExtension.lowercased() == "jpg" {
            guard let (step, index) = parsePhotoFilename(file.lastPathComponent),
                  step >= 13 else { continue }

            if let existing = best[index] {
                if step > existing.step { best[index] = (step, file) }
            } else {
                best[index] = (step, file)
            }
        }

        return best.keys.sorted().compactMap { best[$0]?.url }
    }

    /// Parses filenames like "15_photo_02_final.jpg" → (step: 15, photoIndex: 2).
    private static func parsePhotoFilename(_ name: String) -> (step: Int, index: Int)? {
        let parts = name.components(separatedBy: "_")
        guard parts.count >= 3,
              let step = Int(parts[0]),
              parts[1] == "photo",
              let index = Int(parts[2]) else { return nil }
        return (step, index)
    }

    /// Looks for `<stem>.<ext>` in test-images/ across all supported extensions.
    @MainActor private static func findInputFile(stem: String) -> URL? {
        let exts = ["HEIC", "heic", "DNG", "dng", "JPG", "jpg", "JPEG", "jpeg", "PNG", "png"]
        for ext in exts {
            let candidate = testImagesRoot.appendingPathComponent("\(stem).\(ext)")
            if FileManager.default.fileExists(atPath: candidate.path) { return candidate }
        }
        return nil
    }

    private static func isDirectory(_ url: URL) -> Bool {
        var isDir: ObjCBool = false
        return FileManager.default.fileExists(atPath: url.path, isDirectory: &isDir) && isDir.boolValue
    }
}
