import Foundation

struct SeededRandomNumberGenerator: RandomNumberGenerator {
    private var state: UInt64

    init(seed: UInt64) {
        self.state = seed == 0 ? 0xCAFE_F00D_F00D_BEEF : seed
    }

    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var value = state
        value = (value ^ (value >> 30)) &* 0xBF58_476D_1CE4_E5B9
        value = (value ^ (value >> 27)) &* 0x94D0_49BB_1331_11EB
        return value ^ (value >> 31)
    }

    mutating func nextUnitDouble() -> Double {
        Double(next()) / Double(UInt64.max)
    }

    mutating func nextUniform(in range: ClosedRange<Double>) -> Double {
        range.lowerBound + (range.upperBound - range.lowerBound) * nextUnitDouble()
    }

    mutating func nextGaussian(stdDev: Double) -> Double {
        guard stdDev > 0 else {
            return 0
        }

        let u1 = max(nextUnitDouble(), 1e-12)
        let u2 = nextUnitDouble()
        let magnitude = sqrt(-2.0 * log(u1))
        let angle = 2.0 * Double.pi * u2
        return stdDev * magnitude * cos(angle)
    }
}
