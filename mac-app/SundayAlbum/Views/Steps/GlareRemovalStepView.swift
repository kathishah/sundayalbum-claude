import SwiftUI
import AppKit

/// Before/after glare removal view for a single extracted photo.
/// The tree in StepDetailView handles photo selection; this view shows one photo's result.
struct GlareRemovalStepView: View {
    let job: ProcessingJob
    let photoIndex: Int   // 0-based

    @State private var afterOpacity: Double = 0
    @State private var showGlow = false

    var photo: ExtractedPhoto? { job.extractedPhotos[safe: photoIndex] }

    var body: some View {
        // The step tree already shows the oriented "before" image; show only the
        // glare-removed result here with the reveal animation.
        VStack(spacing: 8) {
            ImagePane(url: photo?.imageURL)
                .shadow(
                    color: showGlow ? Color.saAmber500.opacity(0.2) : .clear,
                    radius: 20
                )
                .opacity(afterOpacity)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding(16)
        .background(Color.saStone900)
        .onAppear { triggerReveal() }
        .onChange(of: photoIndex) { triggerReveal() }
    }

    private func triggerReveal() {
        afterOpacity = 0
        showGlow = false
        Task {
            try? await Task.sleep(for: .milliseconds(80))
            withAnimation(.saReveal) { afterOpacity = 1 }
            withAnimation(.saReveal.delay(0.4)) { showGlow = true }
        }
    }
}

// MARK: - Image pane

private struct ImagePane: View {
    let url: URL?
    @State private var image: NSImage?

    var body: some View {
        Group {
            if let img = image {
                Image(nsImage: img).resizable().aspectRatio(contentMode: .fit)
            } else {
                Rectangle().fill(Color.saStone800)
                    .aspectRatio(4 / 3, contentMode: .fit)
                    .overlay { ProgressView().controlSize(.regular) }
            }
        }
        .clipShape(RoundedRectangle(cornerRadius: 8))
        .task(id: url) {
            image = nil
            if let u = url { image = NSImage(contentsOf: u) }
        }
    }
}

// MARK: - Safe subscript

private extension Array {
    subscript(safe index: Int) -> Element? {
        indices.contains(index) ? self[index] : nil
    }
}
