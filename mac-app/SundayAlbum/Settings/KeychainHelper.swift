import Foundation
import Security

/// Thin wrapper around Security.framework for storing API keys.
enum KeychainHelper {
    private static let service = "com.sundayalbum.mac"

    /// Saves (or overwrites) a string value in the Keychain.
    static func save(key: String, value: String) {
        let data = Data(value.utf8)
        // Delete any existing item first so we can add a fresh one.
        let deleteQuery: [CFString: Any] = [
            kSecClass:       kSecClassGenericPassword,
            kSecAttrService: service,
            kSecAttrAccount: key,
        ]
        SecItemDelete(deleteQuery as CFDictionary)

        let addQuery: [CFString: Any] = [
            kSecClass:       kSecClassGenericPassword,
            kSecAttrService: service,
            kSecAttrAccount: key,
            kSecValueData:   data,
        ]
        SecItemAdd(addQuery as CFDictionary, nil)
    }

    /// Returns the stored string, or nil if absent.
    static func load(key: String) -> String? {
        let query: [CFString: Any] = [
            kSecClass:            kSecClassGenericPassword,
            kSecAttrService:      service,
            kSecAttrAccount:      key,
            kSecReturnData:       true,
            kSecMatchLimit:       kSecMatchLimitOne,
        ]
        var result: AnyObject?
        guard SecItemCopyMatching(query as CFDictionary, &result) == errSecSuccess,
              let data = result as? Data,
              let string = String(data: data, encoding: .utf8) else { return nil }
        return string
    }

    /// Removes an item from the Keychain. Silently ignores missing items.
    static func delete(key: String) {
        let query: [CFString: Any] = [
            kSecClass:       kSecClassGenericPassword,
            kSecAttrService: service,
            kSecAttrAccount: key,
        ]
        SecItemDelete(query as CFDictionary)
    }
}
