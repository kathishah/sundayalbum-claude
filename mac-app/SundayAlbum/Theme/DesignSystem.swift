import SwiftUI

// MARK: - Colors

extension Color {
    // Amber — primary accent
    static let saAmber50  = Color(red: 1.000, green: 0.973, blue: 0.941)
    static let saAmber100 = Color(red: 0.996, green: 0.941, blue: 0.863)
    static let saAmber200 = Color(red: 0.992, green: 0.878, blue: 0.753)
    static let saAmber400 = Color(red: 0.984, green: 0.749, blue: 0.141)
    static let saAmber500 = Color(red: 0.851, green: 0.467, blue: 0.024)
    static let saAmber600 = Color(red: 0.706, green: 0.325, blue: 0.035)
    static let saAmber700 = Color(red: 0.565, green: 0.243, blue: 0.043)

    // Stone — neutrals
    static let saStone50  = Color(red: 0.980, green: 0.980, blue: 0.976)
    static let saStone100 = Color(red: 0.961, green: 0.961, blue: 0.957)
    static let saStone200 = Color(red: 0.906, green: 0.898, blue: 0.894)
    static let saStone400 = Color(red: 0.659, green: 0.635, blue: 0.620)
    static let saStone500 = Color(red: 0.471, green: 0.443, blue: 0.424)
    static let saStone600 = Color(red: 0.341, green: 0.325, blue: 0.306)
    static let saStone700 = Color(red: 0.267, green: 0.251, blue: 0.235)
    static let saStone900 = Color(red: 0.110, green: 0.098, blue: 0.090)

    // Status
    static let saSuccess = Color(red: 0.086, green: 0.639, blue: 0.290)
    static let saError   = Color(red: 0.863, green: 0.149, blue: 0.149)
}

// MARK: - Typography

extension Font {
    static func fraunces(_ size: CGFloat, weight: Font.Weight = .regular) -> Font {
        .custom("Fraunces", size: size).weight(weight)
    }

    static func dmSans(_ size: CGFloat, weight: Font.Weight = .regular) -> Font {
        .custom("DM Sans", size: size).weight(weight)
    }

    static func jetbrainsMono(_ size: CGFloat) -> Font {
        .custom("JetBrains Mono", size: size)
    }
}

// MARK: - Animation

extension Animation {
    /// Standard ease — 200ms, smooth deceleration
    static let saStandard = Animation.timingCurve(0.16, 1, 0.3, 1, duration: 0.2)
    /// Panel slide — 350ms
    static let saSlide    = Animation.timingCurve(0.16, 1, 0.3, 1, duration: 0.35)
    /// Signature glare reveal — 600ms
    static let saReveal   = Animation.timingCurve(0.16, 1, 0.3, 1, duration: 0.6)
    /// Spring for hover/select feedback
    static let saSpring   = Animation.spring(response: 0.4, dampingFraction: 0.6)
}
