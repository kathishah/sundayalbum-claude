import Foundation

extension AppState {
    /// Pre-populated state using real debug/ output images.
    /// Each job is at a different pipeline step to exercise every view.
    static func withMockData() -> AppState {
        let state = AppState()

        func debugDir(_ name: String) -> URL {
            URL(fileURLWithPath: "/Users/dev/dev/sundayalbum-claude/debug/\(name)")
        }
        func testImg(_ name: String) -> URL {
            URL(fileURLWithPath: "/Users/dev/dev/sundayalbum-claude/test-images/\(name)")
        }

        // ── Job 1: harbor — at Page Detection, awaiting user review ──────────
        let job1 = ProcessingJob(
            inputName: "IMG_harbor_normal.HEIC",
            inputURL: testImg("IMG_harbor_normal.HEIC"),
            state: .running,
            currentStep: .pageDetect,
            stepStatus: .awaitingReview
        )
        // No extracted photos yet (split hasn't run)

        // ── Job 2: three_pics — at Glare Removal, 3 photos already split ─────
        let job2 = ProcessingJob(
            inputName: "IMG_three_pics_normal.HEIC",
            inputURL: testImg("IMG_three_pics_normal.HEIC"),
            state: .running,
            currentStep: .glareRemoval,
            stepStatus: .awaitingReview
        )
        let d2 = debugDir("IMG_three_pics_normal")
        job2.extractedPhotos = [
            ExtractedPhoto(
                imageURL: d2.appendingPathComponent("07_photo_01_deglared.jpg"),
                jobInputURL: d2.appendingPathComponent("05b_photo_01_oriented.jpg"),
                jobInputName: job2.inputName, jobID: job2.id
            ),
            ExtractedPhoto(
                imageURL: d2.appendingPathComponent("07_photo_02_deglared.jpg"),
                jobInputURL: d2.appendingPathComponent("05b_photo_02_oriented.jpg"),
                jobInputName: job2.inputName, jobID: job2.id
            ),
            ExtractedPhoto(
                imageURL: d2.appendingPathComponent("07_photo_03_deglared.jpg"),
                jobInputURL: d2.appendingPathComponent("05b_photo_03_oriented.jpg"),
                jobInputName: job2.inputName, jobID: job2.id
            ),
        ]

        // ── Job 3: cave — fully processed ────────────────────────────────────
        let job3 = ProcessingJob(
            inputName: "IMG_cave_normal.HEIC",
            inputURL: testImg("IMG_cave_normal.HEIC"),
            state: .complete,
            currentStep: .done,
            stepStatus: .complete,
            processingTime: 22.1
        )
        let d3 = debugDir("IMG_cave_normal")
        job3.extractedPhotos = [
            ExtractedPhoto(
                imageURL: d3.appendingPathComponent("15_photo_01_final.jpg"),
                jobInputURL: d3.appendingPathComponent("01_loaded.jpg"),
                jobInputName: job3.inputName, jobID: job3.id
            ),
        ]

        // ── Job 4: two_pics — queued ──────────────────────────────────────────
        let job4 = ProcessingJob(
            inputName: "IMG_two_pics_vertical_horizontal_normal.HEIC",
            inputURL: testImg("IMG_two_pics_vertical_horizontal_normal.HEIC"),
            state: .queued,
            currentStep: .load,
            stepStatus: .pending
        )

        state.jobs = [job1, job2, job3, job4]
        return state
    }
}
