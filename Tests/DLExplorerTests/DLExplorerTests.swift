import XCTest
@testable import DLExplorer

final class DLExplorerTests: XCTestCase {
    func testDatasetGenerationIsDeterministic() {
        let lhs = RegressionDataFactory.makeTrainingSamples(noise: 0.05, seed: 123)
        let rhs = RegressionDataFactory.makeTrainingSamples(noise: 0.05, seed: 123)

        XCTAssertEqual(lhs, rhs)
    }

    func testLossFunctionsReturnExpectedValues() {
        XCTAssertEqual(LossKind.mse.value(prediction: 1.5, target: 1.0), 0.25, accuracy: 1e-9)
        XCTAssertEqual(LossKind.mse.derivative(prediction: 1.5, target: 1.0), 1.0, accuracy: 1e-9)

        XCTAssertEqual(LossKind.mae.value(prediction: 1.5, target: 1.0), 0.5, accuracy: 1e-9)
        XCTAssertEqual(LossKind.mae.derivative(prediction: 1.5, target: 1.0), 1.0, accuracy: 1e-9)

        XCTAssertEqual(LossKind.huber.value(prediction: 1.05, target: 1.0), 0.00125, accuracy: 1e-9)
        XCTAssertEqual(LossKind.huber.derivative(prediction: 1.05, target: 1.0), 0.05, accuracy: 1e-9)
    }

    func testOptimizersMoveLossInTheRightDirection() {
        let samples = RegressionDataFactory.makeTrainingSamples(noise: 0, seed: 9, count: 32)

        for optimizer in OptimizerKind.allCases {
            var model = TinyMLP(seed: 5, activation: .tanh)
            var optimizerState = TinyMLP.OptimizerState.make(for: model.layers)
            let config = TrainingConfig(
                optimizer: optimizer,
                loss: .mse,
                activation: .tanh,
                learningRate: optimizer == .adam ? 0.01 : 0.02,
                epochCount: 25,
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
        let samples = RegressionDataFactory.makeTrainingSamples(noise: config.noise, seed: 42)
        var model = TinyMLP(seed: 42 ^ 0xA11C_E5EED, activation: config.activation)
        var optimizerState = TinyMLP.OptimizerState.make(for: model.layers)

        let initialLoss = model.averageLoss(on: samples, kind: config.loss)
        for _ in 0..<500 {
            _ = model.trainEpoch(samples: samples, config: config, optimizerState: &optimizerState)
        }
        let finalLoss = model.averageLoss(on: samples, kind: config.loss)

        XCTAssertLessThan(finalLoss, initialLoss * 0.35)
    }
}
