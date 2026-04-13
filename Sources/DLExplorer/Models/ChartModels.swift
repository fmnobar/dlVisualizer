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

struct EpochSnapshot: Equatable {
    let epoch: Int
    let totalEpochs: Int
    let currentLoss: Double
    let predictionCurve: [SamplePoint]
    let lossHistory: [LossPoint]
}
