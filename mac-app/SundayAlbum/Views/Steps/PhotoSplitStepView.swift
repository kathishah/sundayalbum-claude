import SwiftUI
import AppKit

/// Photo split step view — shows detected regions overlaid on the fitted page image.
/// Each region has 4 independently-draggable corner handles, allowing the user to
/// match keystoned or rotated photo prints (not just axis-aligned rectangles).
/// "Confirm & Re-run" writes the corrected JSON and restarts from photo_split.
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
                    .coordinateSpace(name: "photoSplitCanvas")
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
                                // Convert canvas rect to pixel-space corners [TL, TR, BR, BL]
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

                Text("Drag any corner handle freely to adjust shape. Tap \"Add Region\" then draw to create a new one.")
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
            let label = "Photo \(idx + 1)"

            // Prefer corners key (free-form quad) if present
            if let rawCorners = d["corners"] as? [[Any]], rawCorners.count == 4 {
                let pts: [CGPoint] = rawCorners.compactMap { arr in
                    guard arr.count == 2,
                          let x = (arr[0] as? NSNumber).map({ CGFloat($0.doubleValue) }),
                          let y = (arr[1] as? NSNumber).map({ CGFloat($0.doubleValue) })
                    else { return nil }
                    return CGPoint(x: x, y: y)
                }
                if pts.count == 4 {
                    return EditableRegion(corners: pts, label: label)
                }
            }

            // Fall back to bbox (axis-aligned rectangle)
            guard let raw = d["bbox"] as? [Any], raw.count == 4,
                  let x1 = (raw[0] as? NSNumber).map({ CGFloat($0.doubleValue) }),
                  let y1 = (raw[1] as? NSNumber).map({ CGFloat($0.doubleValue) }),
                  let x2 = (raw[2] as? NSNumber).map({ CGFloat($0.doubleValue) }),
                  let y2 = (raw[3] as? NSNumber).map({ CGFloat($0.doubleValue) })
            else { return nil }
            return EditableRegion(
                pixelRect: CGRect(x: x1, y: y1, width: x2 - x1, height: y2 - y1),
                label: label
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
            let xs = r.corners.map { $0.x }
            let ys = r.corners.map { $0.y }
            return [
                "bbox": [
                    Int((xs.min() ?? 0).rounded()),
                    Int((ys.min() ?? 0).rounded()),
                    Int((xs.max() ?? 0).rounded()),
                    Int((ys.max() ?? 0).rounded()),
                ],
                "corners": r.corners.map { [Int($0.x.rounded()), Int($0.y.rounded())] },
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
    /// Corner points in pixel space: [TL, TR, BR, BL] (clockwise from top-left).
    var corners: [CGPoint]
    var label: String

    /// Initialise from free-form corner points.
    init(corners: [CGPoint], label: String) {
        self.corners = corners
        self.label = label
    }

    /// Convenience initialiser from an axis-aligned rect (used for drawn regions).
    init(pixelRect: CGRect, label: String) {
        self.corners = [
            CGPoint(x: pixelRect.minX, y: pixelRect.minY),
            CGPoint(x: pixelRect.maxX, y: pixelRect.minY),
            CGPoint(x: pixelRect.maxX, y: pixelRect.maxY),
            CGPoint(x: pixelRect.minX, y: pixelRect.maxY),
        ]
        self.label = label
    }
}

// MARK: - Region view (polygon border + 4 independent corner handles)

private struct RegionView: View {
    @Binding var region: EditableRegion
    let canvasSize: CGSize
    let origin: CGPoint
    let scaleX: CGFloat
    let scaleY: CGFloat
    let colour: Color
    let onDelete: () -> Void

    private func canvasPoint(from px: CGPoint) -> CGPoint {
        CGPoint(x: origin.x + px.x * scaleX, y: origin.y + px.y * scaleY)
    }

    private func pixelPoint(from canvas: CGPoint) -> CGPoint {
        CGPoint(x: (canvas.x - origin.x) / scaleX, y: (canvas.y - origin.y) / scaleY)
    }

    private var canvasCorners: [CGPoint] {
        region.corners.map { canvasPoint(from: $0) }
    }

    var body: some View {
        let cc = canvasCorners

        ZStack {
            // Polygon border
            Path { p in
                p.move(to: cc[0])
                p.addLine(to: cc[1])
                p.addLine(to: cc[2])
                p.addLine(to: cc[3])
                p.closeSubpath()
            }
            .stroke(colour, lineWidth: 2)
            .allowsHitTesting(false)

            // Label + delete button near TL corner
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
                        .background(Color.black.opacity(0.01))
                }
                .buttonStyle(.plain)
            }
            .padding(4)
            .position(x: cc[0].x + 52, y: cc[0].y + 14)

            // 4 independent corner handles
            ForEach(0..<4, id: \.self) { idx in
                CornerHandle(position: cc[idx], colour: colour) { canvasPos in
                    let p = pixelPoint(from: canvasPos)
                    var newCorners = region.corners
                    newCorners[idx] = p
                    if isConvex(newCorners) {
                        region.corners = newCorners
                    }
                }
            }
        }
        .frame(width: canvasSize.width, height: canvasSize.height)
    }

    /// Returns true if the 4-point polygon is convex (all cross products same sign).
    private func isConvex(_ pts: [CGPoint]) -> Bool {
        let n = pts.count
        guard n >= 3 else { return false }
        var sign: Int = 0
        for i in 0..<n {
            let a = pts[i], b = pts[(i + 1) % n], c = pts[(i + 2) % n]
            let cross = (b.x - a.x) * (c.y - b.y) - (b.y - a.y) * (c.x - b.x)
            if abs(cross) > 0.01 {
                let s = cross > 0 ? 1 : -1
                if sign == 0 { sign = s }
                else if s != sign { return false }
            }
        }
        return true
    }
}

// MARK: - Corner handle

private struct CornerHandle: View {
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
            // coordinateSpace: .named("photoSplitCanvas") ensures v.location is
            // in the canvas coordinate space, not the handle's 20×20 local frame.
            DragGesture(minimumDistance: 0, coordinateSpace: .named("photoSplitCanvas"))
                .onChanged { v in onDrag(v.location) }
        )
    }
}
