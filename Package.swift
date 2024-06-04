// swift-tools-version:5.5

import PackageDescription

let package = Package(
    name: "Lstsq-CoreML",
    platforms: [
        .iOS(.v14),
        .macOS("11.0")
    ],
    products: [
        .library(
            name: "LstsqCoreML",
            type: .dynamic,
            targets: ["LstsqCoreML"]
        ),
    ],
    targets: [
        .target(
            name: "LstsqCoreML",
            path: "LstsqCoreML"
        )
    ]
)
