import SwiftUI
import AppKit

/// Shows the photo after color correction. For multi-photo jobs, shows a thumbnail strip.
struct ColorCorrectionStepView: View {
    let job: ProcessingJob

    @State private var selectedIndex: Int = 0
    @State private var image: NSImage?
    @State private var brightness: Double = 0.05
    @State private var saturation: Double = 0.15
    @State private var warmth: Double = 0.10
    @State private var sharpness: Double = 0.50
    @State private var showBefore = false

    var photos: [ExtractedPhoto] { job.extractedPhotos }
    var hasMultiple: Bool { photos.count > 1 }
    var displayPhoto: ExtractedPhoto? { photos[safe: selectedIndex] }

    var body: some View {
        VStack(spacing: 0) {
            if hasMultiple {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 8) {
                        ForEach(Array(photos.enumerated()), id: \.offset) { idx, photo in
                            ColorThumbnail(
                                photo: photo,
                                inputName: job.inputName,
                                photoIndex: idx + 1,
                                isSelected: idx == selectedIndex
                            )
                            .onTapGesture {
                                withAnimation(.saStandard) { selectedIndex = idx }
                            }
                        }
                    }
                    .padding(.horizontal, 16)
                    .padding(.vertical, 10)
                }
                .background(Color.saStone100)
                Divider()
            }

            HStack(spacing: 0) {
                // ── Image canvas ────────────────────────────────────────
                ZStack {
                    Color.saStone900

                    if let img = image {
                        Image(nsImage: img)
                            .resizable()
                            .aspectRatio(contentMode: .fit)
                    } else {
                        ProgressView().controlSize(.large)
                    }

                    // Before/after toggle hint
                    VStack {
                        Spacer()
                        HStack {
                            Spacer()
                            Toggle(isOn: $showBefore) {
                                Text(showBefore ? "Before" : "After")
                                    .font(.dmSans(11, weight: .medium))
                                    .foregroundStyle(Color.saStone300)
                            }
                            .toggleStyle(.button)
                            .controlSize(.small)
                            .padding(12)
                        }
                    }
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)

                // ── Controls panel ──────────────────────────────────────
                VStack(alignment: .leading, spacing: 0) {
                    Text("Color Adjustments")
                        .font(.dmSans(13, weight: .semibold))
                        .foregroundStyle(Color.saStone700)
                        .padding(.horizontal, 16)
                        .padding(.top, 20)
                        .padding(.bottom, 16)

                    Divider()

                    ScrollView {
                        VStack(spacing: 20) {
                            AdjustmentSlider(label: "Brightness",
                                             icon: "sun.max",
                                             value: $brightness,
                                             range: -0.5...0.5)
                            AdjustmentSlider(label: "Saturation",
                                             icon: "drop",
                                             value: $saturation,
                                             range: -0.5...0.5)
                            AdjustmentSlider(label: "Warmth",
                                             icon: "thermometer.sun",
                                             value: $warmth,
                                             range: -0.5...0.5)
                            AdjustmentSlider(label: "Sharpness",
                                             icon: "camera.aperture",
                                             value: $sharpness,
                                             range: 0...1)
                        }
                        .padding(16)
                    }

                    Divider()

                    Button("Reset to Defaults") {
                        withAnimation(.saStandard) {
                            brightness = 0.05
                            saturation = 0.15
                            warmth = 0.10
                            sharpness = 0.50
                        }
                    }
                    .buttonStyle(.plain)
                    .font(.dmSans(12))
                    .foregroundStyle(Color.saStone400)
                    .padding(16)
                }
                .frame(width: 220)
                .background(Color.saStone50)
            }
        }
        .task(id: selectedIndex) {
            image = nil
            let url = PipelineStep.colorCorrection.debugImageURL(
                forInputName: job.inputName,
                photoIndex: selectedIndex + 1
            ) ?? displayPhoto?.imageURL ?? job.inputURL
            if let u = url { image = NSImage(contentsOf: u) }
        }
    }
}

// MARK: - Thumbnail

private struct ColorThumbnail: View {
    let photo: ExtractedPhoto
    let inputName: String
    let photoIndex: Int
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
        .task {
            let url = PipelineStep.colorCorrection.debugImageURL(
                forInputName: inputName,
                photoIndex: photoIndex
            ) ?? photo.imageURL
            image = NSImage(contentsOf: url)
        }
    }
}

// MARK: - Slider component

private struct AdjustmentSlider: View {
    let label: String
    let icon: String
    @Binding var value: Double
    let range: ClosedRange<Double>

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Image(systemName: icon)
                    .font(.system(size: 11))
                    .foregroundStyle(Color.saStone400)
                    .frame(width: 16)
                Text(label)
                    .font(.dmSans(12, weight: .medium))
                    .foregroundStyle(Color.saStone700)
                Spacer()
                Text(String(format: "%+.0f%%", value * 100))
                    .font(.jetbrainsMono(11))
                    .foregroundStyle(Color.saStone400)
                    .frame(width: 44, alignment: .trailing)
            }
            Slider(value: $value, in: range)
                .tint(Color.saAmber500)
        }
    }
}

// MARK: - Safe subscript

private extension Array {
    subscript(safe index: Int) -> Element? {
        indices.contains(index) ? self[index] : nil
    }
}
