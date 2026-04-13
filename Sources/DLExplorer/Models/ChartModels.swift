import Foundation

struct SamplePoint: Identifiable, Equatable {
    let x: Double
    let y: Double

    var id: Double { x }
}

struct LossPoint: Identifiable, Equatable {
    let epoch: Int
    let loss: Double

    var id: Int { epoch }
}

struct TrainingExample: Equatable {
    let point: SamplePoint
    let inputs: [Double]
}

struct NetworkNodeSnapshot: Identifiable, Equatable {
    enum Kind: Equatable {
        case feature
        case hidden
        case output
    }

    let id: String
    let label: String
    let value: Double
    let kind: Kind
}

struct NetworkLayerSnapshot: Identifiable, Equatable {
    let id: String
    let title: String
    let nodes: [NetworkNodeSnapshot]
}

struct NetworkConnectionSnapshot: Identifiable, Equatable {
    let id: String
    let fromLayerIndex: Int
    let fromNodeIndex: Int
    let toLayerIndex: Int
    let toNodeIndex: Int
    let weight: Double
}

struct NetworkSnapshot: Equatable {
    let probeX: Double
    let layers: [NetworkLayerSnapshot]
    let connections: [NetworkConnectionSnapshot]
}

struct EpochSnapshot: Equatable {
    let epoch: Int
    let totalEpochs: Int
    let currentLoss: Double
    let predictionCurve: [SamplePoint]
    let lossHistory: [LossPoint]
    let network: NetworkSnapshot
}
