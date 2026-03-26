import SwiftUI
import AppKit

/// Shows the (page-corrected) image with coloured rectangles for each detected photo.
/// The user can confirm the splits or adjust/add regions.
struct PhotoSplitStepView: View {
    let job: ProcessingJob

    @State private var image: NSImage?

    // Mock detected regions as normalized rects
    @State private var regions: [DetectedRegion] = [
        DetectedRegion(id: 0, color: .saAmber500,
                       rect: CGRect(x: 0.03, y: 0.05, width: 0.44, height: 0.88)),
        DetectedRegion(id: 1, color: Color(red: 0.18, green: 0.55, blue: 0.34),
                       rect: CGRect(x: 0.53, y: 0.05, width: 0.44, height: 0.42)),
        DetectedRegion(id: 2, color: Color(red: 0.24, green: 0.48, blue: 0.78),
                       rect: CGRect(x: 0.53, y: 0.53, width: 0.44, height: 0.40)),
    ]

    var body: some View {
        ZStack(alignment: .bottomTrailing) {
            Color.black

            if let img = image {
                Image(nsImage: img)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
            }

            GeometryReader { geo in
                let sz = geo.size
                ForEach(regions) { region in
                    RegionOverlay(region: region, size: sz)
                }
            }

            // "Add region" button
            Button {
                // mock: would enter draw-new-region mode
            } label: {
                Label("Add Region", systemImage: "plus.rectangle.on.rectangle")
                    .font(.dmSans(12, weight: .medium))
            }
            .buttonStyle(.borderedProminent)
            .tint(Color.saAmber500)
            .controlSize(.small)
            .padding(16)
        }
        .task {
            if let url = job.inputURL {
                image = NSImage(contentsOf: url)
            }
        }
    }
}

// MARK: - Region model

private struct DetectedRegion: Identifiable {
    let id: Int
    let color: Color
    var rect: CGRect   // normalized 0–1
}

// MARK: - Region overlay (non-interactive for prototype)

private struct RegionOverlay: View {
    let region: DetectedRegion
    let size: CGSize

    var actual: CGRect {
        CGRect(
            x: region.rect.minX * size.width,
            y: region.rect.minY * size.height,
            width: region.rect.width * size.width,
            height: region.rect.height * size.height
        )
    }

    var body: some View {
        ZStack(alignment: .topLeading) {
            Rectangle()
                .strokeBorder(region.color, lineWidth: 2)
                .frame(width: actual.width, height: actual.height)
                .offset(x: actual.minX, y: actual.minY)

            Text("Photo \(region.id + 1)")
                .font(.dmSans(11, weight: .semibold))
                .foregroundStyle(.white)
                .padding(.horizontal, 6)
                .padding(.vertical, 3)
                .background(region.color)
                .clipShape(RoundedRectangle(cornerRadius: 4))
                .offset(x: actual.minX + 6, y: actual.minY + 6)
        }
    }
}
