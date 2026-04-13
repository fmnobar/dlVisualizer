import AppKit
import SwiftUI

@main
struct DLVisualizerApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate
    @State private var controller = TrainingController()

    var body: some Scene {
        WindowGroup("DL Visualizer", id: "main") {
            ContentView(controller: controller)
                .frame(minWidth: 1120, idealWidth: 1260, minHeight: 760, idealHeight: 840)
        }
        .windowResizability(.contentMinSize)
    }
}

final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)
        NSApp.activate(ignoringOtherApps: true)
    }
}
