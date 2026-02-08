// swift-tools-version: 6.2

import PackageDescription

let package = Package(
    name: "kuyu-world-model",
    platforms: [
        .macOS(.v26)
    ],
    products: [
        .library(
            name: "KuyuWorldModel",
            targets: ["KuyuWorldModel"]
        ),
    ],
    dependencies: [
        .package(path: "../kuyu-core"),
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.29.1"),
    ],
    targets: [
        .target(
            name: "KuyuWorldModel",
            dependencies: [
                .product(name: "KuyuCore", package: "kuyu-core"),
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
            ]
        ),
        .testTarget(
            name: "KuyuWorldModelTests",
            dependencies: ["KuyuWorldModel"]
        ),
    ]
)
