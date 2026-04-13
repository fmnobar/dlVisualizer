import Foundation

enum RegressionDataFactory {
    static let xRange: ClosedRange<Double> = -3.0...3.0
    static let defaultTrainingSampleCount = 96
    static let evaluationPointCount = 181

    static func featureVector(for x: Double, features: [FeatureKind]) -> [Double] {
        features.map { $0.value(at: x) }
    }

    static func makeTrainingExamples(
        target: TargetKind,
        features: [FeatureKind],
        noise: Double,
        seed: UInt64,
        count: Int = defaultTrainingSampleCount
    ) -> [TrainingExample] {
        var generator = SeededRandomNumberGenerator(seed: seed)
        let step = (xRange.upperBound - xRange.lowerBound) / Double(max(count - 1, 1))

        return (0..<count).map { index in
            let x = xRange.lowerBound + Double(index) * step
            let y = target.value(at: x) + generator.nextGaussian(stdDev: noise)
            let point = SamplePoint(x: x, y: y)
            return TrainingExample(point: point, inputs: featureVector(for: x, features: features))
        }
    }

    static func makeEvaluationGrid(count: Int = evaluationPointCount) -> [Double] {
        let step = (xRange.upperBound - xRange.lowerBound) / Double(max(count - 1, 1))
        return (0..<count).map { index in
            xRange.lowerBound + Double(index) * step
        }
    }
}
