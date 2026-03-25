import SwiftUI

@main
struct SundayAlbumApp: App {
    @State private var appState: AppState = {
        #if DEBUG
        // Set MOCK_DATA=1 in the scheme's environment variables to use mock data during UI work.
        if ProcessInfo.processInfo.environment["MOCK_DATA"] == "1" {
            return AppState.withMockData()
        }
        #endif
        return AppState()
    }()

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
