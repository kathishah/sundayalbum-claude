import SwiftUI
import AppKit

// MARK: - Main detail view

struct StepDetailView: View {
    @Environment(AppState.self) private var appState
    @Bindable var job: ProcessingJob

    // The step currently shown in the canvas (may differ from job.currentStep when browsing history)
    @State private var selectedStep: PipelineStep

    init(job: ProcessingJob) {
        self.job = job
        // Start on current step (or last complete step if done)
        let start = job.currentStep == .done ? .colorCorrection : job.currentStep
        self._selectedStep = State(initialValue: start)
    }

    var body: some View {
        VStack(spacing: 0) {
            // ── Breadcrumb ───────────────────────────────────────────
            BreadcrumbBar(crumbs: ["Library", job.inputName]) {
                appState.navigateBack()
            }

            Divider()

            // ── Step strip ───────────────────────────────────────────
            StepStrip(job: job, selectedStep: $selectedStep)
                .frame(height: 94)
                .background(Color.saSurface)

            Divider()

            // ── Step canvas ──────────────────────────────────────────
            Group {
                switch selectedStep {
                case .load, .orientation:
                    DebugImageView(step: selectedStep, inputName: job.inputName)
                case .pageDetect:
                    PageDetectionStepView(job: job)
                case .photoSplit:
                    PhotoSplitStepView(job: job)
                case .glareRemoval:
                    GlareRemovalStepView(job: job)
                case .colorCorrection:
                    ColorCorrectionStepView(job: job)
                case .done:
                    ResultsStepView(job: job)
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .id(selectedStep)

            // ── Action bar ───────────────────────────────────────────
            if selectedStep == job.currentStep && selectedStep.requiresReview {
                // AI result ready — confirm or adjust
                StepActionBar(job: job)
            } else if selectedStep.rawValue < job.currentStep.rawValue && selectedStep != .done {
                // User is browsing a past step — offer to re-process from here
                ReprocessBar(job: job, fromStep: selectedStep)
            }
        }
        .background(Color.saBackground)
    }
}

// MARK: - Breadcrumb

struct BreadcrumbBar: View {
    let crumbs: [String]
    var onBack: () -> Void

    var body: some View {
        HStack(spacing: 6) {
            Button { withAnimation(.saSlide) { onBack() } } label: {
                Image(systemName: "chevron.left")
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(Color.saAmber600)
            }
            .buttonStyle(.plain)

            ForEach(Array(crumbs.enumerated()), id: \.offset) { idx, crumb in
                if idx > 0 {
                    Image(systemName: "chevron.right")
                        .font(.system(size: 9, weight: .medium))
                        .foregroundStyle(Color.saStone400)
                }
                if idx < crumbs.count - 1 {
                    Button(crumb) { withAnimation(.saSlide) { onBack() } }
                        .buttonStyle(.plain)
                        .font(.dmSans(13))
                        .foregroundStyle(Color.saAmber600)
                } else {
                    Text(crumb)
                        .font(.dmSans(13, weight: .semibold))
                        .foregroundStyle(Color.saTextPrimary)
                        .lineLimit(1)
                }
            }
            Spacer()
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 11)
        .background(Color.saSurface)
    }
}

// MARK: - Step strip

struct StepStrip: View {
    let job: ProcessingJob
    @Binding var selectedStep: PipelineStep

    var body: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 0) {
                ForEach(Array(PipelineStep.allCases.enumerated()), id: \.offset) { idx, step in
                    if idx > 0 {
                        // Connector line
                        Rectangle()
                            .fill(job.completedSteps.contains(step)
                                  ? Color.saAmber400 : Color.saStone200)
                            .frame(width: 18, height: 2)
                    }
                    StepTile(
                        step: step,
                        inputName: job.inputName,
                        isSelected: step == selectedStep,
                        isComplete: job.completedSteps.contains(step),
                        isCurrent: step == job.currentStep,
                        isAccessible: job.completedSteps.contains(step) || step == job.currentStep
                    )
                    .onTapGesture {
                        guard job.completedSteps.contains(step) || step == job.currentStep else { return }
                        withAnimation(.saStandard) { selectedStep = step }
                    }
                }
            }
            .padding(.horizontal, 24)
            .padding(.vertical, 14)
        }
    }
}

// MARK: - Step tile

struct StepTile: View {
    let step: PipelineStep
    let inputName: String
    let isSelected: Bool
    let isComplete: Bool
    let isCurrent: Bool
    let isAccessible: Bool

    @State private var thumb: NSImage?

    var body: some View {
        VStack(spacing: 5) {
            ZStack(alignment: .bottomTrailing) {
                // Thumbnail
                Group {
                    if let img = thumb {
                        Image(nsImage: img).resizable().aspectRatio(contentMode: .fill)
                    } else {
                        Rectangle()
                            .fill(Color.saSurface)
                            .overlay {
                                Image(systemName: step.systemImage)
                                    .font(.system(size: 14))
                                    .foregroundStyle(Color.saTextTertiary)
                            }
                    }
                }
                .frame(width: 72, height: 52)
                .clipShape(RoundedRectangle(cornerRadius: 6))
                .overlay {
                    RoundedRectangle(cornerRadius: 6)
                        .strokeBorder(isSelected ? Color.saAmber500 : .clear, lineWidth: 2)
                }
                .shadow(color: isSelected ? Color.saAmber500.opacity(0.35) : .clear, radius: 6)

                // Status badge
                if isComplete {
                    Image(systemName: "checkmark.circle.fill")
                        .font(.system(size: 13))
                        .foregroundStyle(Color.saSuccess)
                        .background(Circle().fill(Color.saCard).padding(2))
                        .offset(x: 4, y: 4)
                } else if isCurrent {
                    Circle()
                        .fill(Color.saAmber500)
                        .frame(width: 8, height: 8)
                        .overlay(Circle().strokeBorder(.white, lineWidth: 1.5))
                        .offset(x: 4, y: 4)
                }
            }

            Text(step.title)
                .font(.dmSans(9, weight: isSelected ? .semibold : .regular))
                .foregroundStyle(isSelected ? Color.saAmber600 : Color.saTextSecondary)
        }
        .opacity(isAccessible ? 1.0 : 0.35)
        .task {
            if let url = step.debugImageURL(forInputName: inputName) {
                thumb = NSImage(contentsOf: url)
            }
        }
    }
}

// MARK: - Action bar (current step awaiting review)

struct StepActionBar: View {
    @Bindable var job: ProcessingJob

    var body: some View {
        VStack(spacing: 0) {
            Divider()
            HStack(spacing: 10) {
                Button {
                    // mock: re-run this step
                } label: {
                    Label("Redo", systemImage: "arrow.counterclockwise")
                        .font(.dmSans(13))
                }
                .buttonStyle(.bordered)
                .controlSize(.regular)

                Spacer()

                Button {
                    // mock: open manual editor
                } label: {
                    Label("Edit Manually", systemImage: "pencil")
                        .font(.dmSans(13))
                }
                .buttonStyle(.bordered)
                .controlSize(.regular)

                Button {
                    withAnimation(.saStandard) { job.advance() }
                } label: {
                    HStack(spacing: 6) {
                        Text("Looks Good")
                            .font(.dmSans(13, weight: .semibold))
                        Image(systemName: "arrow.right")
                            .font(.system(size: 11, weight: .semibold))
                    }
                }
                .buttonStyle(.borderedProminent)
                .tint(Color.saAmber500)
                .controlSize(.regular)
            }
            .padding(.horizontal, 20)
            .padding(.vertical, 14)
        }
        .background(Color.saStone50)
    }
}

// MARK: - Reprocess bar (browsing a past step)

struct ReprocessBar: View {
    @Bindable var job: ProcessingJob
    let fromStep: PipelineStep

    var body: some View {
        VStack(spacing: 0) {
            Divider()
            HStack(spacing: 12) {
                Image(systemName: "info.circle")
                    .foregroundStyle(Color.saAmber500)
                Text("Viewing \(fromStep.title) — make adjustments above")
                    .font(.dmSans(13))
                    .foregroundStyle(Color.saStone500)
                Spacer()
                Button {
                    // mock: reset pipeline to this step and re-run
                    withAnimation(.saStandard) {
                        job.currentStep = fromStep
                        job.stepStatus = fromStep.requiresReview ? .awaitingReview : .running
                        job.state = .running
                        // Drop extracted photos produced after this step
                        if fromStep.rawValue <= PipelineStep.photoSplit.rawValue {
                            job.extractedPhotos = []
                        }
                    }
                } label: {
                    Label("Save & Re-process from Here", systemImage: "arrow.clockwise")
                        .font(.dmSans(13, weight: .semibold))
                }
                .buttonStyle(.borderedProminent)
                .tint(Color.saAmber500)
                .controlSize(.regular)
            }
            .padding(.horizontal, 20)
            .padding(.vertical, 14)
        }
        .background(Color.saStone50)
    }
}

// MARK: - Auto-step view (load, orientation — no user action needed)

struct AutoStepView: View {
    let job: ProcessingJob
    var body: some View {
        VStack(spacing: 16) {
            ProgressView().controlSize(.large)
            Text(job.currentStep.title + "…")
                .font(.dmSans(14))
                .foregroundStyle(Color.saStone500)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color.saStone100)
    }
}

// MARK: - Debug image viewer (for load / orientation steps when browsing history)

struct DebugImageView: View {
    let step: PipelineStep
    let inputName: String
    @State private var image: NSImage?

    var body: some View {
        ZStack {
            Color.saStone900
            if let img = image {
                Image(nsImage: img)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
            } else {
                VStack(spacing: 12) {
                    Image(systemName: step.systemImage)
                        .font(.system(size: 40, weight: .light))
                        .foregroundStyle(Color.saStone600)
                    Text("No debug image for this step")
                        .font(.dmSans(14))
                        .foregroundStyle(Color.saStone500)
                }
            }
        }
        .task {
            if let url = step.debugImageURL(forInputName: inputName) {
                image = NSImage(contentsOf: url)
            }
        }
    }
}
