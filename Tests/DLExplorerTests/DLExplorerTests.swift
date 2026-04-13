import XCTest
@testable import DLExplorer

final class DLExplorerTests: XCTestCase {
    func testDatasetGenerationIsDeterministic() {
        let lhs = RegressionDataFactory.makeTrainingSamples(target: .sineWave, noise: 0.05, seed: 123, count: 48)
        let rhs = RegressionDataFactory.makeTrainingSamples(target: .sineWave, noise: 0.05, seed: 123, count: 48)

        XCTAssertEqual(lhs, rhs)
        XCTAssertEqual(lhs.count, 48)
    }

    func testAlternativeTargetsProduceDistinctCurves() {
        let x = 1.25

        XCTAssertNotEqual(TargetKind.reciprocalBell.value(at: x), TargetKind.sineWave.value(at: x), accuracy: 1e-9)
        XCTAssertNotEqual(TargetKind.cubicCurve.value(at: x), TargetKind.absValley.value(at: x), accuracy: 1e-9)
    }

    func testLossFunctionsReturnExpectedValues() {
        XCTAssertEqual(LossKind.mse.value(prediction: 1.5, target: 1.0), 0.25, accuracy: 1e-9)
        XCTAssertEqual(LossKind.mse.derivative(prediction: 1.5, target: 1.0), 1.0, accuracy: 1e-9)

        XCTAssertEqual(LossKind.mae.value(prediction: 1.5, target: 1.0), 0.5, accuracy: 1e-9)
        XCTAssertEqual(LossKind.mae.derivative(prediction: 1.5, target: 1.0), 1.0, accuracy: 1e-9)

        XCTAssertEqual(LossKind.huber.value(prediction: 1.05, target: 1.0), 0.00125, accuracy: 1e-9)
        XCTAssertEqual(LossKind.huber.derivative(prediction: 1.05, target: 1.0), 0.05, accuracy: 1e-9)

        XCTAssertEqual(LossKind.logCosh.value(prediction: 1.5, target: 1.0), 0.12011450695827745, accuracy: 1e-9)
        XCTAssertEqual(LossKind.logCosh.derivative(prediction: 1.5, target: 1.0), 0.46211715726000974, accuracy: 1e-9)

        XCTAssertEqual(LossKind.cauchy.value(prediction: 1.5, target: 1.0), log(2.0), accuracy: 1e-9)
        XCTAssertEqual(LossKind.cauchy.derivative(prediction: 1.5, target: 1.0), 2.0, accuracy: 1e-9)
    }

    func testOptimizersMoveLossInTheRightDirection() {
        let samples = RegressionDataFactory.makeTrainingSamples(target: .cubicCurve, noise: 0, seed: 9, count: 24)

        for optimizer in OptimizerKind.allCases {
            var model = TinyMLP(seed: 5, activation: .tanh)
            var optimizerState = TinyMLP.OptimizerState.make(for: model.layers)
            let config = TrainingConfig(
                target: .cubicCurve,
                sampleCount: 24,
                seed: 9,
                optimizer: optimizer,
                loss: .mse,
                activation: .tanh,
                learningRate: learningRate(for: optimizer),
                epochCount: 20,
                noise: 0
            )

            let initialLoss = model.averageLoss(on: samples, kind: .mse)
            for _ in 0..<config.epochCount {
                _ = model.trainEpoch(samples: samples, config: config, optimizerState: &optimizerState)
            }
            let finalLoss = model.averageLoss(on: samples, kind: .mse)

            XCTAssertLessThan(finalLoss, initialLoss, "\(optimizer.rawValue) should reduce the loss on a deterministic sample set.")
        }
    }

    func testBaselineConfigurationLearnsTheTargetCurve() {
        let config = TrainingConfig()
        let samples = RegressionDataFactory.makeTrainingSamples(
            target: config.target,
            noise: config.noise,
            seed: config.seed,
            count: config.sampleCount
        )
        var model = TinyMLP(seed: config.seed ^ 0xA11C_E5EED, activation: config.activation)
        var optimizerState = TinyMLP.OptimizerState.make(for: model.layers)

        let initialLoss = model.averageLoss(on: samples, kind: config.loss)
        for _ in 0..<400 {
            _ = model.trainEpoch(samples: samples, config: config, optimizerState: &optimizerState)
        }
        let finalLoss = model.averageLoss(on: samples, kind: config.loss)

        XCTAssertLessThan(finalLoss, initialLoss * 0.4)
    }

    private func learningRate(for optimizer: OptimizerKind) -> Double {
        switch optimizer {
        case .sgd, .momentum, .nesterov:
            return 0.02
        case .rmsProp:
            return 0.01
        case .adaGrad:
            return 0.03
        case .adam:
            return 0.01
        }
    }
}
