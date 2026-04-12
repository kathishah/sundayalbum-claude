import SwiftUI
import AppKit

/// Photo split step view — shows detected regions overlaid on the fitted page image.
/// Each region has 4 draggable corner handles (identical approach to PageDetectionStepView).
/// "Confirm & Re-run" writes the corrected JSON to the debug folder and
/// restarts the pipeline from photo_split onward.
struct PhotoSplitStepView: View {
    let job: ProcessingJob
    @Environment(AppState.self) private var appState

    @State private var pageImage: NSImage?
    @State private var regions: [EditableRegion] = []
    @State private var isDrawing = false
    @State private var drawStart: CGPoint? = nil
    @State private var drawCurrent: CGPoint? = nil
    @State private var isRerunning = false
    @State private var errorMessage: String? = nil
    @State private var loadDebugPath: String? = nil

    private var stem: String {
        URL(fileURLWithPath: job.inputName).deletingPathExtension().lastPathComponent
    }

    var body: some View {
        VStack(spacing: 0) {
            // ── Image canvas ───────────────────────────────────────────
            ZStack {
                Color.black
                GeometryReader { geo in
                    let imgSize = pageImage?.size ?? CGSize(width: 1, height: 1)
                    let fitted  = fitSize(imgSize, in: geo.size)
                    let origin  = CGPoint(
                        x: (geo.size.width  - fitted.width)  / 2,
                        y: (geo.size.height - fitted.height) / 2
                    )
                    let scaleX = imgSize.width  > 0 ? fitted.width  / imgSize.width  : 1
                    let scaleY = imgSize.height > 0 ? fitted.height / imgSize.height : 1

                    ZStack(alignment: .topLeading) {

                        // Background image
                        if let img = pageImage {
                            Image(nsImage: img)
                                .resizable()
                                .frame(width: fitted.width, height: fitted.height)
                                .position(x: origin.x + fitted.width / 2,
                                          y: origin.y + fitted.height / 2)
                        } else if let debugPath = loadDebugPath {
                            VStack(spacing: 8) {
                                Image(systemName: "photo.badge.exclamationmark")
                                    .font(.system(size: 32))
                                    .foregroundStyle(Color.saTextSecondary)
                                Text("Could not load image")
                                    .font(.dmSans(13, weight: .semibold))
                                    .foregroundStyle(Color.saTextSecondary)
                                Text(debugPath)
                                    .font(.system(size: 10, design: .monospaced))
                                    .foregroundStyle(Color.saTextTertiary)
                                    .multilineTextAlignment(.center)
                                    .padding(.horizontal, 20)
                            }
                            .frame(maxWidth: .infinity, maxHeight: .infinity)
                        }

                        // Region overlays — one per detected/edited region
                        ForEach($regions) { $region in
                            RegionView(
                                region: $region,
                                canvasSize: geo.size,
                                origin: origin,
                                scaleX: scaleX,
                                scaleY: scaleY,
                                colour: paletteColour(for: region),
                                onDelete: { removeRegion(id: region.id) }
                            )
                            .allowsHitTesting(!isDrawing)
                        }

                        // In-progress draw rectangle
                        if isDrawing, let s = drawStart, let c = drawCurrent {
                            let r = rectFrom(a: s, b: c)
                            Path { p in p.addRect(r) }
                                .stroke(Color.saAmber500,
                                        style: StrokeStyle(lineWidth: 2, dash: [6]))
                                .allowsHitTesting(false)
                        }
                    }
                    .frame(width: geo.size.width, height: geo.size.height)
                    .contentShape(Rectangle())
                    .gesture(
                        DragGesture(minimumDistance: 4)
                            .onChanged { v in
                                guard isDrawing else { return }
                                if drawStart == nil { drawStart = v.startLocation }
                                drawCurrent = v.location
                            }
                            .onEnded { v in
                                guard isDrawing, let s = drawStart else { return }
                                let r = rectFrom(a: s, b: v.location)
                                let px = CGRect(
                                    x: (r.minX - origin.x) / scaleX,
                                    y: (r.minY - origin.y) / scaleY,
                                    width:  r.width  / scaleX,
                                    height: r.height / scaleY
                                )
                                if px.width > 10 && px.height > 10 {
                                    let newRegion = EditableRegion(
                                        pixelRect: px,
                                        label: "Photo \(regions.count + 1)"
                                    )
                                    regions.append(newRegion)
                                }
                                drawStart   = nil
                                drawCurrent = nil
                                isDrawing   = false
                            }
                    )
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .task { loadState() }
            .task(id: job.currentStep.rawValue) { loadState() }

            Divider()

            // ── Footer ─────────────────────────────────────────────────
            VStack(alignment: .leading, spacing: 10) {
                HStack(spacing: 10) {
                    Button {
                        isDrawing = true
                    } label: {
                        Label("Add Region", systemImage: "plus.rectangle.on.rectangle")
                            .font(.dmSans(12, weight: .medium))
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                    .tint(isDrawing ? Color.saAmber500 : nil)

                    Text("\(regions.count) region\(regions.count == 1 ? "" : "s")")
                        .font(.dmSans(12))
                        .foregroundStyle(Color.saTextSecondary)

                    Spacer()

                    if let err = errorMessage {
                        Text(err)
                            .font(.dmSans(11))
                            .foregroundStyle(Color.saError)
                    }

                    Button {
                        Task { await confirmAndRerun() }
                    } label: {
                        HStack(spacing: 6) {
                            if isRerunning { ProgressView().controlSize(.mini) }
                            Text(isRerunning ? "Starting…" : "Confirm & Re-run from Here")
                                .font(.dmSans(13, weight: .semibold))
                            if !isRerunning {
                                Image(systemName: "arrow.clockwise")
                                    .font(.system(size: 11, weight: .semibold))
                            }
                        }
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(Color.saAmber500)
                    .controlSize(.regular)
                    .disabled(isRerunning || regions.isEmpty)
                }

                Text("Drag corner handles to resize a region. Tap \"Add Region\" then draw to create a new one.")
                    .font(.dmSans(11))
                    .foregroundStyle(Color.saTextTertiary)
            }
            .padding(16)
            .background(Color.saCard)
        }
    }

    // MARK: - Load

    @MainActor
    private func loadState() {
        let debugDir = effectiveDebugFolder()
        let s = stem
        print("[PhotoSplitStepView] loadState — stem: \(s), debugDir: \(debugDir.path)")

        var loaded = false
        for name in ["\(s)_04_photo_boundaries.jpg", "\(s)_03_page_warped.jpg"] {
            let url = debugDir.appendingPathComponent(name)
            let exists = FileManager.default.fileExists(atPath: url.path)
            print("[PhotoSplitStepView]   trying \(name) — exists: \(exists)")
            guard exists else { continue }
            if let data = try? Data(contentsOf: url), let img = NSImage(data: data) {
                print("[PhotoSplitStepView]   loaded image size: \(img.size)")
                pageImage = img
                loadDebugPath = nil
                loaded = true
                break
            } else {
                print("[PhotoSplitStepView]   FAILED to create NSImage from \(url.path)")
            }
        }
        if !loaded && pageImage == nil {
            let tried = debugDir.appendingPathComponent("\(s)_04_photo_boundaries.jpg").path
            print("[PhotoSplitStepView]   image load failed — showing error for \(tried)")
            loadDebugPath = tried
        }

        let jsonURL = debugDir.appendingPathComponent("\(s)_05_photo_detections.json")
        print("[PhotoSplitStepView]   json path: \(jsonURL.path)")
        guard let data = try? Data(contentsOf: jsonURL),
              let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let dets = root["detections"] as? [[String: Any]] else {
            print("[PhotoSplitStepView]   json load FAILED")
            return
        }
        print("[PhotoSplitStepView]   json loaded \(dets.count) detection(s)")

        regions = dets.enumerated().compactMap { idx, d in
            guard let raw = d["bbox"] as? [Any], raw.count == 4,
                  let x1 = (raw[0] as? NSNumber).map({ CGFloat($0.doubleValue) }),
                  let y1 = (raw[1] as? NSNumber).map({ CGFloat($0.doubleValue) }),
                  let x2 = (raw[2] as? NSNumber).map({ CGFloat($0.doubleValue) }),
                  let y2 = (raw[3] as? NSNumber).map({ CGFloat($0.doubleValue) })
            else { return nil }
            return EditableRegion(
                pixelRect: CGRect(x: x1, y: y1, width: x2 - x1, height: y2 - y1),
                label: "Photo \(idx + 1)"
            )
        }
    }

    // MARK: - Confirm & re-run

    @MainActor
    private func confirmAndRerun() async {
        isRerunning = true
        errorMessage = nil
        defer { isRerunning = false }

        let dets: [[String: Any]] = regions.map { r in
            [
                "bbox": [
                    Int(r.pixelRect.minX.rounded()),
                    Int(r.pixelRect.minY.rounded()),
                    Int(r.pixelRect.maxX.rounded()),
                    Int(r.pixelRect.maxY.rounded()),
                ],
                "confidence": 1.0,
                "region_type": "photo",
                "orientation": "unknown",
            ]
        }
        let payload: [String: Any] = [
            "photo_count": dets.count,
            "multi_blob": false,
            "detections": dets,
        ]

        let debugDir = effectiveDebugFolder()
        let jsonURL = debugDir.appendingPathComponent("\(stem)_05_photo_detections.json")
        do {
            try FileManager.default.createDirectory(at: debugDir, withIntermediateDirectories: true)
            let data = try JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted])
            try data.write(to: jsonURL)
        } catch {
            errorMessage = "Could not write detections: \(error.localizedDescription)"
            return
        }

        let runner = PipelineRunner(job: job)
        runner.reprocessFromPhotoSplit()
    }

    // MARK: - Helpers

    private func removeRegion(id: UUID) {
        regions.removeAll { $0.id == id }
        for i in regions.indices { regions[i].label = "Photo \(i + 1)" }
    }

    private func effectiveDebugFolder() -> URL {
        let s = AppSettings.shared
        return s.debugOutputEnabled
            ? s.debugFolder
            : RuntimeManager.shared.cliWorkingDirectory.appendingPathComponent("debug")
    }

    private func fitSize(_ imgSize: CGSize, in container: CGSize) -> CGSize {
        guard imgSize.width > 0, imgSize.height > 0,
              container.width > 0, container.height > 0 else { return container }
        let scale = min(container.width / imgSize.width, container.height / imgSize.height)
        return CGSize(width: imgSize.width * scale, height: imgSize.height * scale)
    }

    private func rectFrom(a: CGPoint, b: CGPoint) -> CGRect {
        CGRect(x: min(a.x, b.x), y: min(a.y, b.y),
               width: abs(b.x - a.x), height: abs(b.y - a.y))
    }

    private let palette: [Color] = [
        .saAmber500,
        Color(red: 0.18, green: 0.55, blue: 0.34),
        Color(red: 0.24, green: 0.48, blue: 0.78),
        Color(red: 0.80, green: 0.25, blue: 0.25),
        Color(red: 0.55, green: 0.25, blue: 0.78),
    ]
    private func paletteColour(for region: EditableRegion) -> Color {
        palette[abs(region.label.hashValue) % palette.count]
    }
}

// MARK: - Region model

struct EditableRegion: Identifiable {
    let id = UUID()
    var pixelRect: CGRect   // image pixel coordinates
    var label: String
}

// MARK: - Region view (border + 4 corner handles)
//
// Mirrors PageDetectionStepView's CornerHandle pattern exactly:
// • Canvas-sized so Path coordinates equal canvas coordinates.
// • Corner handles use .position() — hit area IS where the dot appears.
// • DragGesture uses v.location (canvas space) — no base tracking needed.

private struct RegionView: View {
    @Binding var region: EditableRegion
    let canvasSize: CGSize
    let origin: CGPoint
    let scaleX: CGFloat
    let scaleY: CGFloat
    let colour: Color
    let onDelete: () -> Void

    // Canvas positions of the 4 corners
    private var posTL: CGPoint { canvasPoint(px: region.pixelRect.minX, py: region.pixelRect.minY) }
    private var posTR: CGPoint { canvasPoint(px: region.pixelRect.maxX, py: region.pixelRect.minY) }
    private var posBR: CGPoint { canvasPoint(px: region.pixelRect.maxX, py: region.pixelRect.maxY) }
    private var posBL: CGPoint { canvasPoint(px: region.pixelRect.minX, py: region.pixelRect.maxY) }

    private func canvasPoint(px: CGFloat, py: CGFloat) -> CGPoint {
        CGPoint(x: origin.x + px * scaleX, y: origin.y + py * scaleY)
    }

    private func pixelPoint(from canvas: CGPoint) -> CGPoint {
        CGPoint(x: (canvas.x - origin.x) / scaleX, y: (canvas.y - origin.y) / scaleY)
    }

    var body: some View {
        ZStack {
            // Rectangle border (non-interactive, visual only)
            Path { p in
                p.move(to: posTL)
                p.addLine(to: posTR)
                p.addLine(to: posBR)
                p.addLine(to: posBL)
                p.closeSubpath()
            }
            .stroke(colour, lineWidth: 2)
            .allowsHitTesting(false)

            // Label + delete button
            HStack(spacing: 4) {
                Text(region.label)
                    .font(.dmSans(11, weight: .semibold))
                    .foregroundStyle(.white)
                    .padding(.horizontal, 6).padding(.vertical, 3)
                    .background(colour)
                    .clipShape(RoundedRectangle(cornerRadius: 4))
                Button { onDelete() } label: {
                    Image(systemName: "xmark.circle.fill")
                        .font(.system(size: 13))
                        .foregroundStyle(colour)
                        .background(Color.black.opacity(0.01))   // widen tap target
                }
                .buttonStyle(.plain)
            }
            .padding(4)
            // Place label just inside the top-left corner
            .position(x: posTL.x + 52, y: posTL.y + 14)

            // TL handle — dragging changes minX, minY
            RectCornerHandle(position: posTL, colour: colour) { cp in
                let p = pixelPoint(from: cp)
                let newW = region.pixelRect.maxX - p.x
                let newH = region.pixelRect.maxY - p.y
                if newW > 20 && newH > 20 {
                    region.pixelRect = CGRect(x: p.x, y: p.y, width: newW, height: newH)
                }
            }

            // TR handle — dragging changes maxX, minY
            RectCornerHandle(position: posTR, colour: colour) { cp in
                let p = pixelPoint(from: cp)
                let newW = p.x - region.pixelRect.minX
                let newH = region.pixelRect.maxY - p.y
                if newW > 20 && newH > 20 {
                    region.pixelRect = CGRect(x: region.pixelRect.minX, y: p.y,
                                              width: newW, height: newH)
                }
            }

            // BR handle — dragging changes maxX, maxY
            RectCornerHandle(position: posBR, colour: colour) { cp in
                let p = pixelPoint(from: cp)
                let newW = p.x - region.pixelRect.minX
                let newH = p.y - region.pixelRect.minY
                if newW > 20 && newH > 20 {
                    region.pixelRect = CGRect(x: region.pixelRect.minX, y: region.pixelRect.minY,
                                              width: newW, height: newH)
                }
            }

            // BL handle — dragging changes minX, maxY
            RectCornerHandle(position: posBL, colour: colour) { cp in
                let p = pixelPoint(from: cp)
                let newW = region.pixelRect.maxX - p.x
                let newH = p.y - region.pixelRect.minY
                if newW > 20 && newH > 20 {
                    region.pixelRect = CGRect(x: p.x, y: region.pixelRect.minY,
                                              width: newW, height: newH)
                }
            }
        }
        // Canvas-sized so all .position() coordinates are in canvas space
        .frame(width: canvasSize.width, height: canvasSize.height)
    }
}

// MARK: - Corner handle (same visual style as PageDetectionStepView.CornerHandle)

private struct RectCornerHandle: View {
    let position: CGPoint
    let colour: Color
    /// Called with the new canvas position while dragging.
    let onDrag: (CGPoint) -> Void

    var body: some View {
        ZStack {
            Circle()
                .fill(colour)
                .frame(width: 20, height: 20)
            Circle()
                .strokeBorder(Color.white, lineWidth: 2)
                .frame(width: 20, height: 20)
        }
        .shadow(color: .black.opacity(0.4), radius: 4)
        .position(position)        // .position() moves the hit area; .offset() does NOT
        .gesture(
            DragGesture(minimumDistance: 0)
                .onChanged { v in onDrag(v.location) }
        )
    }
}
