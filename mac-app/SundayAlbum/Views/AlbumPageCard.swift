import SwiftUI
import AppKit

// MARK: - Compact card (library grid)

struct AlbumPageCard: View {
    let job: ProcessingJob
    @State private var beforeImage: NSImage?

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {

            // ── Thumbnail row — centered ─────────────────────────────
            HStack(alignment: .center, spacing: 10) {
                Spacer(minLength: 0)

                ThumbBox(image: beforeImage)
                    .frame(width: 60, height: 88)

                Image(systemName: "arrow.right")
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundStyle(Color.saAmber400)

                AfterSection(job: job, thumbHeight: 88, maxVisible: 1)

                Spacer(minLength: 0)
            }
            .padding(12)

            Divider()

            // ── Info row ─────────────────────────────────────────────
            Text(job.inputName)
                .font(.dmSans(12, weight: .semibold))
                .foregroundStyle(Color.saStone700)
                .lineLimit(1)
                .truncationMode(.middle)
                .frame(maxWidth: .infinity, alignment: .center)
                .padding(.horizontal, 12)
                .padding(.vertical, 10)
        }
        .background(Color.white)
        .clipShape(RoundedRectangle(cornerRadius: 12))
        .shadow(color: Color.saStone900.opacity(0.07), radius: 6, y: 2)
        .task { beforeImage = job.loadBeforeImage() }
    }
}

// MARK: - Expanded overlay card (single-click)

struct ExpandedAlbumCard: View {
    let job: ProcessingJob
    var onViewDetails: () -> Void

    @State private var beforeImage: NSImage?

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {

            // ── Large thumbnail row — no labels ──────────────────────
            HStack(alignment: .center, spacing: 16) {
                Spacer(minLength: 0)

                ThumbBox(image: beforeImage)
                    .frame(width: 120, height: 160)

                Image(systemName: "arrow.right")
                    .font(.system(size: 16, weight: .semibold))
                    .foregroundStyle(Color.saAmber500)

                AfterSection(job: job, thumbHeight: 160)

                Spacer(minLength: 0)
            }
            .padding(20)

            Divider()

            // ── Footer ───────────────────────────────────────────────
            HStack(spacing: 12) {
                VStack(alignment: .leading, spacing: 5) {
                    Text(job.inputName)
                        .font(.dmSans(14, weight: .semibold))
                        .foregroundStyle(Color.saStone700)
                        .lineLimit(1)
                    JobStatusLine(job: job)
                }
                Spacer()
                Button("View Step Details") {
                    withAnimation(.saSlide) { onViewDetails() }
                }
                .buttonStyle(.borderedProminent)
                .tint(Color.saAmber500)
                .font(.dmSans(13, weight: .semibold))
                .controlSize(.regular)
            }
            .padding(20)
        }
        .background(Color.white)
        .clipShape(RoundedRectangle(cornerRadius: 16))
        .shadow(color: Color.saStone900.opacity(0.22), radius: 32, y: 12)
        .task { beforeImage = job.loadBeforeImage() }
    }
}

// MARK: - ThumbBox (before — always portrait crop, no label)

struct ThumbBox: View {
    let image: NSImage?

    var body: some View {
        Group {
            if let img = image {
                Image(nsImage: img)
                    .resizable()
                    .aspectRatio(contentMode: .fill)
            } else {
                Rectangle()
                    .fill(Color.saStone200)
                    .overlay { ProgressView().controlSize(.small) }
            }
        }
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }
}

// MARK: - AfterSection

struct AfterSection: View {
    let job: ProcessingJob
    let thumbHeight: CGFloat
    /// Max number of thumbnails to show before collapsing into a "+N" badge.
    var maxVisible: Int = 3

    var body: some View {
        if job.state == .complete, !job.extractedPhotos.isEmpty {
            // Fully processed — show output thumbnails, capped at maxVisible
            let visible = Array(job.extractedPhotos.prefix(maxVisible))
            let overflow = job.extractedPhotos.count - visible.count
            HStack(spacing: thumbHeight > 100 ? 8 : 5) {
                ForEach(visible) { photo in
                    AfterThumb(url: photo.imageURL, height: thumbHeight)
                }
                if overflow > 0 {
                    ZStack {
                        RoundedRectangle(cornerRadius: 6).fill(Color.saStone200)
                        Text("+\(overflow)")
                            .font(.dmSans(12, weight: .semibold))
                            .foregroundStyle(Color.saStone500)
                    }
                    .frame(width: thumbHeight * 0.65, height: thumbHeight)
                }
            }
        } else {
            // In progress or queued — show pie-chart progress wheel
            PipelineProgressWheel(job: job, size: thumbHeight)
        }
    }
}

// MARK: - AfterThumb (natural portrait / landscape aspect ratio)

struct AfterThumb: View {
    let url: URL
    let height: CGFloat
    @State private var image: NSImage?

    /// Width derived from actual image aspect ratio, capped at 3:2 landscape
    private var thumbWidth: CGFloat {
        guard let img = image else { return height * 0.75 }  // default portrait while loading
        let ratio = img.size.width / img.size.height
        return height * min(ratio, 1.5)
    }

    var body: some View {
        Group {
            if let img = image {
                Image(nsImage: img)
                    .resizable()
                    .aspectRatio(contentMode: .fill)
            } else {
                Rectangle().fill(Color.saStone200)
            }
        }
        .frame(width: thumbWidth, height: height)
        .clipShape(RoundedRectangle(cornerRadius: 6))
        .animation(.saStandard, value: thumbWidth)
        .task { image = NSImage(contentsOf: url) }
    }
}

// MARK: - PipelineProgressWheel (replaces blank processing placeholder)

struct PipelineProgressWheel: View {
    let job: ProcessingJob
    let size: CGFloat

    private let totalSteps = PipelineStep.allCases.count - 1  // 6 real steps (exclude .done)

    private var completedCount: Int {
        min(job.currentStep.rawValue, totalSteps)
    }

    var body: some View {
        ZStack {
            // Pie segments — one per step
            ForEach(0..<totalSteps, id: \.self) { i in
                PieSegment(index: i, total: totalSteps)
                    .fill(i < completedCount ? Color.saAmber500 : Color.saStone200)
            }

            // Donut hole
            Circle()
                .fill(Color.white)
                .padding(size * 0.22)

            // Center label
            VStack(spacing: 1) {
                Text("\(completedCount)")
                    .font(.dmSans(size * 0.22, weight: .bold))
                    .foregroundStyle(Color.saStone700)
                Text("of \(totalSteps)")
                    .font(.dmSans(size * 0.12))
                    .foregroundStyle(Color.saStone400)
            }
        }
        .frame(width: size, height: size)
    }
}

struct PieSegment: Shape {
    let index: Int
    let total: Int
    private let gapDegrees: Double = 3

    func path(in rect: CGRect) -> Path {
        let center = CGPoint(x: rect.midX, y: rect.midY)
        let radius = min(rect.width, rect.height) / 2
        let slice = 360.0 / Double(total)
        let start = Angle.degrees(Double(index) * slice - 90 + gapDegrees / 2)
        let end   = Angle.degrees(Double(index + 1) * slice - 90 - gapDegrees / 2)

        var path = Path()
        path.move(to: center)
        path.addArc(center: center, radius: radius,
                    startAngle: start, endAngle: end, clockwise: false)
        path.closeSubpath()
        return path
    }
}

// MARK: - JobStatusLine

struct JobStatusLine: View {
    let job: ProcessingJob
    private let totalSteps = PipelineStep.allCases.count - 1

    var body: some View {
        switch job.state {
        case .queued:
            Label("Queued", systemImage: "clock")
                .font(.dmSans(11))
                .foregroundStyle(Color.saStone400)

        case .running:
            HStack(spacing: 6) {
                // Slim custom bar — avoids NSProgressIndicator layout warnings
                GeometryReader { geo in
                    ZStack(alignment: .leading) {
                        Capsule().fill(Color.saStone200)
                        Capsule().fill(Color.saAmber500)
                            .frame(width: geo.size.width * job.progressFraction)
                    }
                }
                .frame(width: 52, height: 3)

                let stepNum = min(job.currentStep.rawValue + 1, totalSteps)
                let stepLabel = job.currentStepName ?? job.currentStep.title
                Text("Step \(stepNum) of \(totalSteps): \(stepLabel)")
                    .font(.dmSans(11))
                    .foregroundStyle(
                        job.stepStatus == .awaitingReview ? Color.saAmber600 : Color.saStone500
                    )
            }

        case .complete:
            Label(
                "\(job.extractedPhotos.count) photo\(job.extractedPhotos.count == 1 ? "" : "s") extracted"
                    + (job.processingTime.map { " · \(String(format: "%.1f", $0))s" } ?? ""),
                systemImage: "checkmark.circle.fill"
            )
            .font(.dmSans(11))
            .foregroundStyle(Color.saSuccess)

        case .failed:
            Label(job.errorMessage ?? "Failed", systemImage: "exclamationmark.circle.fill")
                .font(.dmSans(11))
                .foregroundStyle(Color.saError)
        }
    }
}

// MARK: - Helpers

private extension ProcessingJob {
    /// Use debug/01_loaded.jpg if available (faster), else fall back to original HEIC
    func loadBeforeImage() -> NSImage? {
        if let url = PipelineStep.load.debugImageURL(forInputName: inputName) {
            return NSImage(contentsOf: url)
        }
        return inputURL.flatMap { NSImage(contentsOf: $0) }
    }
}
