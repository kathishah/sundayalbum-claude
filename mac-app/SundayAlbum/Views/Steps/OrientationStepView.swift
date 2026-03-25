import SwiftUI
import AppKit

/// Orientation review step — shows the AI-corrected orientation for each extracted photo
/// and lets the user override the rotation and scene description before re-triggering.
struct OrientationStepView: View {
    @Bindable var job: ProcessingJob

    @State private var selectedIndex: Int = 0

    var photos: [ExtractedPhoto] { job.extractedPhotos }
    var hasMultiple: Bool { photos.count > 1 }

    var body: some View {
        VStack(spacing: 0) {
            if hasMultiple {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 8) {
                        ForEach(Array(photos.enumerated()), id: \.offset) { idx, photo in
                            OrientationThumbnail(
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

            if let photo = photos[safe: selectedIndex] {
                OrientationPhotoPanel(
                    photo: photo,
                    inputName: job.inputName,
                    photoIndex: selectedIndex + 1
                )
            } else {
                OrientationPlaceholder()
            }
        }
    }
}

// MARK: - Thumbnail strip item

private struct OrientationThumbnail: View {
    let photo: ExtractedPhoto
    let inputName: String
    let photoIndex: Int
    let isSelected: Bool

    @State private var image: NSImage?

    var body: some View {
        VStack(spacing: 4) {
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
            .rotationEffect(rotationAngle)

            if photo.rotationOverride != nil {
                Image(systemName: "pencil.circle.fill")
                    .font(.system(size: 10))
                    .foregroundStyle(Color.saAmber500)
            }
        }
        .task {
            let url = PipelineStep.orientation.debugImageURL(forInputName: inputName, photoIndex: photoIndex)
                ?? photo.imageURL
            image = NSImage(contentsOf: url)
        }
    }

    var rotationAngle: Angle {
        .degrees(Double(photo.rotationOverride ?? 0))
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

                // Rotation picker
                VStack(alignment: .leading, spacing: 8) {
                    Text("Rotation")
                        .font(.dmSans(12, weight: .semibold))
                        .foregroundStyle(Color.saTextSecondary)

                    RotationPicker(selected: $pendingRotation)
                        .onChange(of: pendingRotation) { checkDirty() }
                }

                // Scene description
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

                    Text("Passed to glare removal as scene context. Leave blank to use the AI-generated description.")
                        .font(.dmSans(11))
                        .foregroundStyle(Color.saTextTertiary)
                }

                Spacer()

                // Action buttons
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
            // Load oriented debug image if available, fall back to output image
            let url = PipelineStep.orientation.debugImageURL(forInputName: inputName, photoIndex: photoIndex)
                ?? photo.imageURL
            image = NSImage(contentsOf: url)

            // Sync pending state from photo model
            pendingRotation = photo.rotationOverride ?? 0
            pendingDescription = photo.sceneDescription ?? ""
            isDirty = false
        }
    }

    private func checkDirty() {
        let savedRotation = photo.rotationOverride ?? 0
        let savedDesc = photo.sceneDescription ?? ""
        isDirty = pendingRotation != savedRotation || pendingDescription != savedDesc
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
            ForEach(options, id: \.degrees) { option in
                Button {
                    withAnimation(.saStandard) { selected = option.degrees }
                } label: {
                    HStack(spacing: 4) {
                        if option.degrees != 0 {
                            Image(systemName: "rotate.right")
                                .font(.system(size: 10))
                        }
                        Text(option.label)
                            .font(.dmSans(12, weight: selected == option.degrees ? .semibold : .regular))
                    }
                    .padding(.horizontal, 8)
                    .padding(.vertical, 5)
                    .background(
                        selected == option.degrees
                            ? Color.saAmber500
                            : Color.saCard
                    )
                    .foregroundStyle(
                        selected == option.degrees
                            ? Color.white
                            : Color.saTextPrimary
                    )
                    .clipShape(RoundedRectangle(cornerRadius: 6))
                    .overlay {
                        RoundedRectangle(cornerRadius: 6)
                            .strokeBorder(
                                selected == option.degrees ? Color.clear : Color.saStone200,
                                lineWidth: 1
                            )
                    }
                }
                .buttonStyle(.plain)
            }
        }
    }
}

// MARK: - Placeholder when no photos extracted yet

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
