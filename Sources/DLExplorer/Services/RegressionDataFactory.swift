import Foundation

enum RegressionDataFactory {
    static let xRange: ClosedRange<Double> = -3.0...3.0
    static let defaultTrainingSampleCount = 96
    static let evaluationPointCount = 181

    static func makeTrainingSamples(
        target: TargetKind,
        noise: Double,
        seed: UInt64,
        count: Int = defaultTrainingSampleCount
    ) -> [SamplePoint] {
        var generator = SeededRandomNumberGenerator(seed: seed)
        let step = (xRange.upperBound - xRange.lowerBound) / Double(max(count - 1, 1))

        return (0..<count).map { index in
            let x = xRange.lowerBound + Double(index) * step
            let y = target.value(at: x) + generator.nextGaussian(stdDev: noise)
            return SamplePoint(x: x, y: y)
        }
    }

    static func makeEvaluationGrid(count: Int = evaluationPointCount) -> [Double] {
        let step = (xRange.upperBound - xRange.lowerBound) / Double(max(count - 1, 1))
        return (0..<count).map { index in
            xRange.lowerBound + Double(index) * step
        }
    }
}
