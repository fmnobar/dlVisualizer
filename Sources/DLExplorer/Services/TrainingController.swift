import Foundation
import Observation

@MainActor
@Observable
final class TrainingController {
    private(set) var targetSamples: [SamplePoint] = []
    private(set) var latestSnapshot: EpochSnapshot?
    private(set) var isTraining = false
    var config: TrainingConfig

    let architecture = "[1, 32, 32, 1]"

    private let evaluationXs = RegressionDataFactory.makeEvaluationGrid()
    private var scheduledRestartTask: Task<Void, Never>?
    private var trainingTask: Task<Void, Never>?
    private var runToken = UUID()

    var targetEquation: String {
        config.target.equation
    }

    init(config: TrainingConfig = TrainingConfig()) {
        self.config = config
        restartTraining(immediate: true)
    }

    func updateConfig(_ mutate: (inout TrainingConfig) -> Void) {
        mutate(&config)
        config.sampleCount = min(max(config.sampleCount, TrainingConfig.sampleCountRange.lowerBound), TrainingConfig.sampleCountRange.upperBound)
        config.epochCount = min(max(config.epochCount, TrainingConfig.epochRange.lowerBound), TrainingConfig.epochRange.upperBound)
        restartTraining()
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
        let targetSamples = RegressionDataFactory.makeTrainingSamples(
            target: activeConfig.target,
            noise: activeConfig.noise,
            seed: activeConfig.seed,
            count: activeConfig.sampleCount
        )
        let initialSeed = activeConfig.seed ^ 0xA11C_E5EED
        let publishDelay = Duration.milliseconds(18)

        self.targetSamples = targetSamples
        self.latestSnapshot = nil
        self.isTraining = true

        trainingTask = Task.detached(priority: .userInitiated) { [evaluationXs] in
            var model = TinyMLP(seed: initialSeed, activation: activeConfig.activation)
            var optimizerState = TinyMLP.OptimizerState.make(for: model.layers)

            var history: [LossPoint] = []
            let initialLoss = model.averageLoss(on: targetSamples, kind: activeConfig.loss)
            history.append(LossPoint(epoch: 0, loss: initialLoss))

            let initialSnapshot = EpochSnapshot(
                epoch: 0,
                totalEpochs: activeConfig.epochCount,
                currentLoss: initialLoss,
                predictionCurve: model.predictionCurve(xs: evaluationXs),
                lossHistory: history
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

                    let loss = model.trainEpoch(samples: targetSamples, config: activeConfig, optimizerState: &optimizerState)
                    history.append(LossPoint(epoch: epoch, loss: loss))

                    if Self.shouldPublish(epoch: epoch, totalEpochs: activeConfig.epochCount) {
                        let snapshot = EpochSnapshot(
                            epoch: epoch,
                            totalEpochs: activeConfig.epochCount,
                            currentLoss: loss,
                            predictionCurve: model.predictionCurve(xs: evaluationXs),
                            lossHistory: history
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
                    predictionCurve: model.predictionCurve(xs: evaluationXs),
                    lossHistory: history
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
}
