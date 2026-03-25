import SwiftUI
import AppKit

/// Side-by-side before/after view for glare removal review.
/// For multi-photo jobs, shows a thumbnail strip so the user can review each photo.
struct GlareRemovalStepView: View {
    let job: ProcessingJob

    @State private var selectedIndex: Int = 0
    @State private var afterOpacity: Double = 0
    @State private var showGlow = false

    var photos: [ExtractedPhoto] { job.extractedPhotos }
    var hasMultiple: Bool { photos.count > 1 }

    var body: some View {
        VStack(spacing: 0) {
            if hasMultiple {
                // Thumbnail strip for multi-photo jobs
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 8) {
                        ForEach(Array(photos.enumerated()), id: \.offset) { idx, photo in
                            ThumbnailStrip(photo: photo, isSelected: idx == selectedIndex)
                                .onTapGesture {
                                    withAnimation(.saStandard) {
                                        selectedIndex = idx
                                        afterOpacity = 0
                                        showGlow = false
                                    }
                                    triggerReveal()
                                }
                        }
                    }
                    .padding(.horizontal, 16)
                    .padding(.vertical, 10)
                }
                .background(Color.saStone100)
                Divider()
            }

            // Before / After canvas
            HStack(spacing: 0) {
                // Before
                VStack(spacing: 8) {
                    Text("BEFORE")
                        .font(.dmSans(10, weight: .semibold))
                        .foregroundStyle(Color.saStone400)
                        .tracking(1.5)
                    ImagePane(url: photos[safe: selectedIndex]?.jobInputURL)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .padding(16)

                Divider()

                // After
                VStack(spacing: 8) {
                    Text("AFTER")
                        .font(.dmSans(10, weight: .semibold))
                        .foregroundStyle(Color.saAmber500)
                        .tracking(1.5)
                    ImagePane(url: photos[safe: selectedIndex]?.imageURL)
                        .shadow(
                            color: showGlow ? Color.saAmber500.opacity(0.2) : .clear,
                            radius: 20
                        )
                        .opacity(afterOpacity)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .padding(16)
            }
            .background(Color.saStone900)
        }
        .onAppear { triggerReveal() }
        .onChange(of: selectedIndex) { triggerReveal() }
    }

    func triggerReveal() {
        afterOpacity = 0
        showGlow = false
        Task {
            try? await Task.sleep(for: .milliseconds(80))
            withAnimation(.saReveal) { afterOpacity = 1 }
            withAnimation(.saReveal.delay(0.4)) { showGlow = true }
        }
    }
}

// MARK: - Sub-views

private struct ThumbnailStrip: View {
    let photo: ExtractedPhoto
    let isSelected: Bool
    @State private var image: NSImage?

    var body: some View {
        Group {
            if let img = image {
                Image(nsImage: img).resizable().aspectRatio(contentMode: .fill)
            } else {
                Rectangle().fill(Color.saStone200)
            }
        }
        .frame(width: 56, height: 56)
        .clipShape(RoundedRectangle(cornerRadius: 6))
        .overlay {
            RoundedRectangle(cornerRadius: 6)
                .strokeBorder(isSelected ? Color.saAmber500 : Color.clear, lineWidth: 2)
        }
        .task { image = NSImage(contentsOf: photo.imageURL) }
    }
}

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

// Needed for the dark image pane background
private extension Color {
    static let saStone800 = Color(red: 0.165, green: 0.149, blue: 0.137)
}
