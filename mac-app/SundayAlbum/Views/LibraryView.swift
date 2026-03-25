import SwiftUI
import UniformTypeIdentifiers

struct LibraryView: View {
    @Environment(AppState.self) private var appState
    @State private var isDropTargeted = false

    let columns = [GridItem(.adaptive(minimum: 280, maximum: 400), spacing: 16)]

    var body: some View {
        ZStack {
            // ── Scrollable grid ──────────────────────────────────────────────
            ScrollView {
                VStack(alignment: .leading, spacing: 0) {
                    // Header
                    HStack(alignment: .center) {
                        Text("Library")
                            .font(.fraunces(32, weight: .semibold))
                            .foregroundStyle(Color.saStone900)
                        Spacer()
                        Button {
                            let urls = FileImporter.openPanel()
                            appState.addFiles(urls)
                        } label: {
                            Label("Add Photos", systemImage: "plus")
                                .font(.dmSans(13, weight: .medium))
                        }
                        .buttonStyle(.borderedProminent)
                        .tint(Color.saAmber500)
                        .controlSize(.regular)
                    }
                    .padding(.horizontal, 32)
                    .padding(.top, 32)
                    .padding(.bottom, 24)

                    if appState.jobs.isEmpty {
                        DropZoneView()
                            .frame(height: 320)
                            .padding(.horizontal, 32)
                    } else {
                        LazyVGrid(columns: columns, spacing: 16) {
                            ForEach(appState.jobs) { job in
                                AlbumPageCard(job: job)
                                    .opacity(appState.expandedJobID != nil && job.id != appState.expandedJobID ? 0.3 : 1.0)
                                    .scaleEffect(appState.expandedJobID != nil && job.id != appState.expandedJobID ? 0.95 : 1.0)
                                    .animation(.saSpring, value: appState.expandedJobID)
                                    .onTapGesture(count: 2) {
                                        withAnimation(.saSlide) {
                                            appState.navigate(to: .stepDetail(jobID: job.id))
                                        }
                                    }
                                    .onTapGesture(count: 1) {
                                        withAnimation(.saSpring) {
                                            appState.expandedJobID = (appState.expandedJobID == job.id) ? nil : job.id
                                        }
                                    }
                            }
                        }
                        .padding(.horizontal, 32)
                        .padding(.bottom, 40)
                    }
                }
            }
            .blur(radius: appState.expandedJobID != nil ? 3 : 0)
            .animation(.saStandard, value: appState.expandedJobID)

            // ── Expanded card overlay ────────────────────────────────────────
            if let jobID = appState.expandedJobID,
               let job = appState.jobFor(id: jobID) {

                // Dim backdrop — tap to dismiss
                Color.black.opacity(0.5)
                    .ignoresSafeArea()
                    .onTapGesture {
                        withAnimation(.saSpring) { appState.expandedJobID = nil }
                    }

                ExpandedAlbumCard(job: job) {
                    appState.navigate(to: .stepDetail(jobID: job.id))
                }
                .frame(maxWidth: 640)
                .padding(48)
                .transition(.scale(scale: 0.94).combined(with: .opacity))
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color.saStone100)
        // Drop target — active even when grid is populated (not just on DropZoneView)
        .onDrop(of: [UTType.fileURL], isTargeted: $isDropTargeted) { providers in
            handleDrop(providers)
        }
        .overlay {
            if isDropTargeted {
                RoundedRectangle(cornerRadius: 0)
                    .strokeBorder(Color.saAmber500, lineWidth: 3)
                    .ignoresSafeArea()
                    .allowsHitTesting(false)
            }
        }
        .animation(.saStandard, value: isDropTargeted)
    }

    private func handleDrop(_ providers: [NSItemProvider]) -> Bool {
        guard !providers.isEmpty else { return false }
        Task { @MainActor in
            var dropped: [URL] = []
            for provider in providers {
                if let url = await provider.loadFileURL() {
                    dropped.append(url)
                }
            }
            let resolved = FileImporter.resolveURLs(dropped)
            appState.addFiles(resolved)
        }
        return true
    }
}
