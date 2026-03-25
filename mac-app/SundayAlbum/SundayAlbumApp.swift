import SwiftUI

@main
struct SundayAlbumApp: App {
    @State private var appState = AppState.withMockData()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environment(appState)
        }
        .windowStyle(.titleBar)
        .windowToolbarStyle(.unified)
        .defaultSize(width: 1280, height: 760)
    }
}
