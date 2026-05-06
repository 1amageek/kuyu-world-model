import KuyuCore

/// Root-body analytical state used to bridge EnvironmentStep snapshots into a
/// WorldModelProtocol without depending on a concrete physics package.
public struct RootBodyAnalyticalState: AnalyticalState {
    public static let dimensionCount = 13

    public let snapshot: PlantStateSnapshot

    public init(snapshot: PlantStateSnapshot) {
        self.snapshot = snapshot
    }

    public init(observation: EnvironmentObservation) {
        self.init(snapshot: observation.plantState)
    }

    public var dimensions: Int { Self.dimensionCount }

    public func toArray() -> [Float] {
        let root = snapshot.root
        return [
            Float(root.position.x), Float(root.position.y), Float(root.position.z),
            Float(root.velocity.x), Float(root.velocity.y), Float(root.velocity.z),
            Float(root.orientation.w), Float(root.orientation.x),
            Float(root.orientation.y), Float(root.orientation.z),
            Float(root.angularVelocity.x), Float(root.angularVelocity.y),
            Float(root.angularVelocity.z),
        ]
    }

    public func toPlantStateSnapshot() -> PlantStateSnapshot {
        snapshot
    }
}
