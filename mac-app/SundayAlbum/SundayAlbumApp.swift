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

    private let appSettings = AppSettings.shared

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environment(appState)
                .environment(appSettings)
                .task {
                    // Silently re-validate saved keys on every launch so status
                    // resolves before the user tries to process anything.
                    if appSettings.anthropicKey() != nil {
                        await appSettings.testAnthropicKey()
                    }
                    if appSettings.openaiKey() != nil {
                        await appSettings.testOpenAIKey()
                    }
                }
        }
        .windowStyle(.titleBar)
        .windowToolbarStyle(.unified)
        .defaultSize(width: 1280, height: 760)

        // ⌘, opens this automatically on macOS
        Settings {
            SettingsView()
                .environment(appSettings)
        }
    }
}
