import SwiftUI
import AppKit

/// Shows the photo after color correction with interactive sliders.
/// Sliders are mock — values don't feed back into the image in the prototype.
struct ColorCorrectionStepView: View {
    let job: ProcessingJob

    @State private var image: NSImage?
    @State private var brightness: Double = 0.05
    @State private var saturation: Double = 0.15
    @State private var warmth: Double = 0.10
    @State private var sharpness: Double = 0.50
    @State private var showBefore = false

    var displayPhoto: ExtractedPhoto? { job.extractedPhotos.first }

    var body: some View {
        HStack(spacing: 0) {
            // ── Image canvas ────────────────────────────────────────
            ZStack {
                Color.saStone900

                if let img = image {
                    Image(nsImage: img)
                        .resizable()
                        .aspectRatio(contentMode: .fit)
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
        .task {
            if let url = displayPhoto?.imageURL ?? job.inputURL {
                image = NSImage(contentsOf: url)
            }
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

private extension Color {
    static let saStone300 = Color(red: 0.792, green: 0.780, blue: 0.769)
    static let saStone800 = Color(red: 0.165, green: 0.149, blue: 0.137)
}
