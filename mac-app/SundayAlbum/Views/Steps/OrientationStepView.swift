import SwiftUI
import AppKit

/// Orientation review for a single extracted photo.
/// The tree in StepDetailView handles photo selection; this view shows one photo's controls.
struct OrientationStepView: View {
    @Bindable var job: ProcessingJob
    let photoIndex: Int   // 0-based

    var photo: ExtractedPhoto? { job.extractedPhotos[safe: photoIndex] }

    var body: some View {
        if let photo {
            OrientationPhotoPanel(
                photo: photo,
                inputName: job.inputName,
                photoIndex: photoIndex + 1
            )
        } else {
            OrientationPlaceholder()
        }
    }
}

// MARK: - Per-photo control panel

private struct OrientationPhotoPanel: View {
    @Bindable var photo: ExtractedPhoto
    let inputName: String
    let photoIndex: Int

    @State private var image: NSImage?
    @State private var pendingRotation: Int = 0
    @State private var pendingDescription: String = ""
    @State private var isDirty = false

    var body: some View {
        HStack(spacing: 0) {
            // ── Image canvas ──────────────────────────────────────────
            ZStack {
                Color.saStone900
                if let img = image {
                    Image(nsImage: img)
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                        .rotationEffect(.degrees(Double(pendingRotation)))
                        .animation(.saStandard, value: pendingRotation)
                        .padding(24)
                } else {
                    ProgressView().controlSize(.large)
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)

            Divider()

            // ── Controls panel ────────────────────────────────────────
            VStack(alignment: .leading, spacing: 20) {
                Text("Orientation")
                    .font(.fraunces(18))
                    .foregroundStyle(Color.saTextPrimary)

                VStack(alignment: .leading, spacing: 8) {
                    Text("Rotation")
                        .font(.dmSans(12, weight: .semibold))
                        .foregroundStyle(Color.saTextSecondary)
                    RotationPicker(selected: $pendingRotation)
                        .onChange(of: pendingRotation) { checkDirty() }
                }

                VStack(alignment: .leading, spacing: 8) {
                    Text("Scene description")
                        .font(.dmSans(12, weight: .semibold))
                        .foregroundStyle(Color.saTextSecondary)

                    TextEditor(text: $pendingDescription)
                        .font(.dmSans(13))
                        .scrollContentBackground(.hidden)
                        .background(Color.saCard)
                        .clipShape(RoundedRectangle(cornerRadius: 6))
                        .overlay {
                            RoundedRectangle(cornerRadius: 6)
                                .strokeBorder(Color.saStone200, lineWidth: 1)
                        }
                        .frame(minHeight: 80, maxHeight: 120)
                        .onChange(of: pendingDescription) { checkDirty() }

                    Text("Passed to glare removal as scene context. Leave blank to use the AI description.")
                        .font(.dmSans(11))
                        .foregroundStyle(Color.saTextTertiary)
                }

                Spacer()

                if isDirty {
                    HStack(spacing: 8) {
                        Button("Discard") {
                            pendingRotation = photo.rotationOverride ?? 0
                            pendingDescription = photo.sceneDescription ?? ""
                            isDirty = false
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.regular)

                        Button("Apply") {
                            photo.rotationOverride = pendingRotation == 0 ? nil : pendingRotation
                            photo.sceneDescription = pendingDescription.isEmpty ? nil : pendingDescription
                            isDirty = false
                        }
                        .buttonStyle(.borderedProminent)
                        .tint(Color.saAmber500)
                        .controlSize(.regular)
                    }
                }
            }
            .padding(20)
            .frame(width: 260)
            .frame(maxHeight: .infinity)
            .background(Color.saBackground)
        }
        .task(id: photoIndex) {
            let url = PipelineStep.orientation.debugImageURL(
                forInputName: inputName, photoIndex: photoIndex
            ) ?? photo.imageURL
            image = NSImage(contentsOf: url)
            pendingRotation = photo.rotationOverride ?? 0
            pendingDescription = photo.sceneDescription ?? ""
            isDirty = false
        }
    }

    private func checkDirty() {
        isDirty = pendingRotation != (photo.rotationOverride ?? 0)
            || pendingDescription != (photo.sceneDescription ?? "")
    }
}

// MARK: - Rotation picker

private struct RotationPicker: View {
    @Binding var selected: Int

    private let options: [(label: String, degrees: Int)] = [
        ("0°",   0),
        ("90°",  90),
        ("180°", 180),
        ("270°", 270),
    ]

    var body: some View {
        HStack(spacing: 6) {
            ForEach(options, id: \.degrees) { opt in
                Button {
                    withAnimation(.saStandard) { selected = opt.degrees }
                } label: {
                    HStack(spacing: 4) {
                        if opt.degrees != 0 {
                            Image(systemName: "rotate.right")
                                .font(.system(size: 10))
                        }
                        Text(opt.label)
                            .font(.dmSans(12, weight: selected == opt.degrees ? .semibold : .regular))
                    }
                    .padding(.horizontal, 8)
                    .padding(.vertical, 5)
                    .background(selected == opt.degrees ? Color.saAmber500 : Color.saCard)
                    .foregroundStyle(selected == opt.degrees ? Color.white : Color.saTextPrimary)
                    .clipShape(RoundedRectangle(cornerRadius: 6))
                    .overlay {
                        RoundedRectangle(cornerRadius: 6)
                            .strokeBorder(
                                selected == opt.degrees ? Color.clear : Color.saStone200,
                                lineWidth: 1
                            )
                    }
                }
                .buttonStyle(.plain)
            }
        }
    }
}

// MARK: - Placeholder

private struct OrientationPlaceholder: View {
    var body: some View {
        VStack(spacing: 12) {
            Image(systemName: "rotate.right")
                .font(.system(size: 40, weight: .light))
                .foregroundStyle(Color.saStone600)
            Text("Waiting for photos to be extracted…")
                .font(.dmSans(14))
                .foregroundStyle(Color.saStone500)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color.saStone900)
    }
}

// MARK: - Safe subscript

private extension Array {
    subscript(safe index: Int) -> Element? {
        indices.contains(index) ? self[index] : nil
    }
}
