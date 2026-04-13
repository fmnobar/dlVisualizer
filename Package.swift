// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "codex_dl",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .executable(
            name: "DLVisualizer",
            targets: ["DLVisualizer"]
        )
    ],
    targets: [
        .executableTarget(
            name: "DLVisualizer",
            path: "Sources/DLExplorer"
        ),
        .testTarget(
            name: "DLVisualizerTests",
            dependencies: ["DLVisualizer"],
            path: "Tests/DLExplorerTests"
        )
    ]
)
