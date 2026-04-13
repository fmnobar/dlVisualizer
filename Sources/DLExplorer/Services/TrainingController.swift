import Foundation
import Observation

@MainActor
@Observable
final class TrainingController {
    private(set) var targetSamples: [SamplePoint] = []
    private(set) var latestSnapshot: EpochSnapshot?
    private(set) var isTraining = false
    var config: TrainingConfig

    private let evaluationXs = RegressionDataFactory.makeEvaluationGrid()
    private var scheduledRestartTask: Task<Void, Never>?
    private var trainingTask: Task<Void, Never>?
    private var runToken = UUID()

    var targetEquation: String {
        config.target.equation
    }

    var architecture: String {
        let widths = [config.features.count] + config.activeHiddenLayerSizes + [1]
        return widths.description
    }

    init(config: TrainingConfig = TrainingConfig()) {
        self.config = config
        restartTraining(immediate: true)
    }

    func updateConfig(_ mutate: (inout TrainingConfig) -> Void) {
        mutate(&config)
        config.features = FeatureKind.orderedSelection(from: config.features)
        if config.features.isEmpty {
            config.features = [.x]
        }
        config.sampleCount = min(max(config.sampleCount, TrainingConfig.sampleCountRange.lowerBound), TrainingConfig.sampleCountRange.upperBound)
        config.epochCount = min(max(config.epochCount, TrainingConfig.epochRange.lowerBound), TrainingConfig.epochRange.upperBound)
        config.hiddenLayerCount = min(max(config.hiddenLayerCount, TrainingConfig.hiddenLayerCountRange.lowerBound), TrainingConfig.hiddenLayerCountRange.upperBound)
        for index in config.hiddenLayerSizes.indices {
            config.hiddenLayerSizes[index] = min(max(config.hiddenLayerSizes[index], TrainingConfig.hiddenLayerSizeRange.lowerBound), TrainingConfig.hiddenLayerSizeRange.upperBound)
        }
        restartTraining()
    }

    func setFeature(_ feature: FeatureKind, enabled: Bool) {
        updateConfig { config in
            var selected = Set(config.features)
            if enabled {
                selected.insert(feature)
            } else {
                selected.remove(feature)
            }
            config.features = FeatureKind.allCases.filter { selected.contains($0) }
        }
    }

    func setHiddenLayerCount(_ count: Int) {
        updateConfig { $0.hiddenLayerCount = count }
    }

    func setHiddenLayerSize(_ size: Int, at index: Int) {
        updateConfig { config in
            guard config.hiddenLayerSizes.indices.contains(index) else {
                return
            }
            config.hiddenLayerSizes[index] = size
        }
    }

    func rerun() {
        restartTraining(immediate: true)
    }

    func resetSeed() {
        config.seed &+= 1
        restartTraining(immediate: true)
    }

    private func restartTraining(immediate: Bool = false) {
        scheduledRestartTask?.cancel()

        if immediate {
            startTraining()
            return
        }

        scheduledRestartTask = Task { [weak self] in
            try? await Task.sleep(for: .milliseconds(220))
            guard !Task.isCancelled else {
                return
            }
            await MainActor.run {
                self?.startTraining()
            }
        }
    }

    private func startTraining() {
        trainingTask?.cancel()
        runToken = UUID()

        let activeToken = runToken
        let activeConfig = config
        let trainingExamples = RegressionDataFactory.makeTrainingExamples(
            target: activeConfig.target,
            features: activeConfig.features,
            noise: activeConfig.noise,
            seed: activeConfig.seed,
            count: activeConfig.sampleCount
        )
        let targetSamples = trainingExamples.map(\.point)
        let initialSeed = activeConfig.seed ^ 0xA11C_E5EED
        let publishDelay = Duration.milliseconds(18)
        let widths = [activeConfig.features.count] + activeConfig.activeHiddenLayerSizes + [1]

        self.targetSamples = targetSamples
        self.latestSnapshot = nil
        self.isTraining = true

        trainingTask = Task.detached(priority: .userInitiated) { [evaluationXs] in
            var model = TinyMLP(seed: initialSeed, activation: activeConfig.activation, widths: widths)
            var optimizerState = TinyMLP.OptimizerState.make(for: model.layers)
            let probeX = Self.visualizationProbeX

            var history: [LossPoint] = []
            let initialLoss = model.averageLoss(on: trainingExamples, kind: activeConfig.loss)
            history.append(LossPoint(epoch: 0, loss: initialLoss))

            let initialSnapshot = EpochSnapshot(
                epoch: 0,
                totalEpochs: activeConfig.epochCount,
                currentLoss: initialLoss,
                predictionCurve: model.predictionCurve(xs: evaluationXs, features: activeConfig.features),
                lossHistory: history,
                network: model.makeNetworkSnapshot(
                    features: activeConfig.features,
                    hiddenLayerSizes: activeConfig.activeHiddenLayerSizes,
                    probeX: probeX
                )
            )

            await MainActor.run { [weak self] in
                guard let self, self.runToken == activeToken else {
                    return
                }
                self.latestSnapshot = initialSnapshot
            }

            do {
                for epoch in 1...activeConfig.epochCount {
                    try Task.checkCancellation()

                    let loss = model.trainEpoch(examples: trainingExamples, config: activeConfig, optimizerState: &optimizerState)
                    history.append(LossPoint(epoch: epoch, loss: loss))

                    if Self.shouldPublish(epoch: epoch, totalEpochs: activeConfig.epochCount) {
                        let snapshot = EpochSnapshot(
                            epoch: epoch,
                            totalEpochs: activeConfig.epochCount,
                            currentLoss: loss,
                            predictionCurve: model.predictionCurve(xs: evaluationXs, features: activeConfig.features),
                            lossHistory: history,
                            network: model.makeNetworkSnapshot(
                                features: activeConfig.features,
                                hiddenLayerSizes: activeConfig.activeHiddenLayerSizes,
                                probeX: probeX
                            )
                        )

                        await MainActor.run { [weak self] in
                            guard let self, self.runToken == activeToken else {
                                return
                            }
                            self.latestSnapshot = snapshot
                        }

                        try await Task.sleep(for: publishDelay)
                    }
                }

                let finalLoss = history.last?.loss ?? initialLoss
                let finalSnapshot = EpochSnapshot(
                    epoch: activeConfig.epochCount,
                    totalEpochs: activeConfig.epochCount,
                    currentLoss: finalLoss,
                    predictionCurve: model.predictionCurve(xs: evaluationXs, features: activeConfig.features),
                    lossHistory: history,
                    network: model.makeNetworkSnapshot(
                        features: activeConfig.features,
                        hiddenLayerSizes: activeConfig.activeHiddenLayerSizes,
                        probeX: probeX
                    )
                )

                await MainActor.run { [weak self] in
                    guard let self, self.runToken == activeToken else {
                        return
                    }
                    self.latestSnapshot = finalSnapshot
                    self.isTraining = false
                }
            } catch {
                await MainActor.run { [weak self] in
                    guard let self, self.runToken == activeToken else {
                        return
                    }
                    self.isTraining = false
                }
            }
        }
    }

    nonisolated private static func shouldPublish(epoch: Int, totalEpochs: Int) -> Bool {
        switch epoch {
        case totalEpochs:
            return true
        case 1...40:
            return true
        case 41...240:
            return epoch.isMultiple(of: 4)
        case 241...720:
            return epoch.isMultiple(of: 8)
        default:
            return epoch.isMultiple(of: 16)
        }
    }

    nonisolated private static let visualizationProbeX = 0.75
}
