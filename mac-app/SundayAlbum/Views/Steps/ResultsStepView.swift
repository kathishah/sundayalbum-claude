import SwiftUI
import AppKit

/// Final step — grid of extracted photos + inline before/after panel.
struct ResultsStepView: View {
    @Environment(AppState.self) private var appState
    let job: ProcessingJob

    @State private var selectedPhotoID: UUID?

    let columns = [GridItem(.adaptive(minimum: 160, maximum: 200), spacing: 10)]

    var selectedPhoto: ExtractedPhoto? {
        job.extractedPhotos.first { $0.id == selectedPhotoID }
            ?? job.extractedPhotos.first
    }

    var body: some View {
        HStack(spacing: 0) {
            // ── Photo grid ──────────────────────────────────────────
            VStack(spacing: 0) {
                HStack {
                    Text("\(job.extractedPhotos.count) photo\(job.extractedPhotos.count == 1 ? "" : "s") extracted")
                        .font(.dmSans(13, weight: .medium))
                        .foregroundStyle(Color.saStone700)
                    Spacer()
                    Button("Export All") {
                        ExportActions.exportToFolder(job.extractedPhotos)
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(Color.saAmber500)
                    .controlSize(.small)
                    .font(.dmSans(12, weight: .medium))
                }
                .padding(.horizontal, 16)
                .padding(.vertical, 12)

                Divider()

                ScrollView {
                    LazyVGrid(columns: columns, spacing: 10) {
                        ForEach(job.extractedPhotos) { photo in
                            ResultThumbnail(
                                photo: photo,
                                isSelected: photo.id == selectedPhotoID
                            )
                            .onTapGesture {
                                withAnimation(.saStandard) {
                                    selectedPhotoID = photo.id
                                }
                            }
                        }
                    }
                    .padding(12)
                }
                .background(Color.saStone50)
            }
            .frame(maxWidth: .infinity)

            Divider()

            // ── Inline comparison ───────────────────────────────────
            if let photo = selectedPhoto {
                ComparisonView(photo: photo)
                    .frame(width: 320)
            }
        }
    }
}

// MARK: - Thumbnail

private struct ResultThumbnail: View {
    let photo: ExtractedPhoto
    let isSelected: Bool

    @State private var image: NSImage?
    @State private var isHovered = false

    var body: some View {
        ZStack(alignment: .bottom) {
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
            .frame(height: 140)
            .clipped()

            if isHovered {
                HStack(spacing: 14) {
                    Button {
                        ExportActions.showInFinder(photo.imageURL)
                    } label: {
                        Image(systemName: "folder")
                            .font(.system(size: 12, weight: .medium))
                    }
                    .buttonStyle(.plain)
                    Button {
                        Task { await ExportActions.addToPhotos(url: photo.imageURL) }
                    } label: {
                        Image(systemName: "photo.badge.plus")
                            .font(.system(size: 12, weight: .medium))
                    }
                    .buttonStyle(.plain)
                }
                .foregroundStyle(.white)
                .padding(.vertical, 8)
                .frame(maxWidth: .infinity)
                .background(.ultraThinMaterial)
                .transition(.move(edge: .bottom).combined(with: .opacity))
            }
        }
        .frame(maxWidth: .infinity)
        .background(Color.white)
        .clipShape(RoundedRectangle(cornerRadius: 8))
        .overlay {
            RoundedRectangle(cornerRadius: 8)
                .strokeBorder(
                    isSelected ? Color.saAmber500 : Color.saStone200,
                    lineWidth: isSelected ? 2 : 1
                )
        }
        .shadow(
            color: isSelected ? Color.saAmber500.opacity(0.2) : Color.saStone900.opacity(0.07),
            radius: isSelected ? 10 : 2,
            y: isSelected ? 0 : 1
        )
        .scaleEffect(isHovered && !isSelected ? 1.02 : 1.0)
        .animation(.saStandard, value: isHovered)
        .onHover { isHovered = $0 }
        .task { image = NSImage(contentsOf: photo.imageURL) }
    }
}
