// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "codex_dl",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .executable(
            name: "DLExplorer",
            targets: ["DLExplorer"]
        )
    ],
    targets: [
        .executableTarget(
            name: "DLExplorer",
            path: "Sources/DLExplorer"
        ),
        .testTarget(
            name: "DLExplorerTests",
            dependencies: ["DLExplorer"],
            path: "Tests/DLExplorerTests"
        )
    ]
)
