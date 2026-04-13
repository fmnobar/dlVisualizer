import Foundation

enum RegressionDataFactory {
    static let xRange: ClosedRange<Double> = -3.0...3.0
    static let trainingSampleCount = 96
    static let evaluationPointCount = 181

    static func targetFunction(_ x: Double) -> Double {
        1.0 / (1.0 + x * x)
    }

    static func makeTrainingSamples(noise: Double, seed: UInt64, count: Int = trainingSampleCount) -> [SamplePoint] {
        var generator = SeededRandomNumberGenerator(seed: seed)
        let step = (xRange.upperBound - xRange.lowerBound) / Double(max(count - 1, 1))

        return (0..<count).map { index in
            let x = xRange.lowerBound + Double(index) * step
            let y = targetFunction(x) + generator.nextGaussian(stdDev: noise)
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
