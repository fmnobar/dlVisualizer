import Foundation

enum TargetKind: String, CaseIterable, Identifiable {
    case reciprocalBell = "Reciprocal Bell"
    case sineWave = "Sine Wave"
    case cubicCurve = "Cubic Curve"
    case absValley = "Abs Valley"

    var id: Self { self }

    var equation: String {
        switch self {
        case .reciprocalBell:
            return "1 / (1 + x^2)"
        case .sineWave:
            return "sin(x)"
        case .cubicCurve:
            return "0.08x^3 - 0.35x"
        case .absValley:
            return "0.38|x| - 0.35"
        }
    }

    func value(at x: Double) -> Double {
        switch self {
        case .reciprocalBell:
            return 1.0 / (1.0 + x * x)
        case .sineWave:
            return sin(x)
        case .cubicCurve:
            return (0.08 * x * x * x) - (0.35 * x)
        case .absValley:
            return (0.38 * abs(x)) - 0.35
        }
    }
}

enum OptimizerKind: String, CaseIterable, Identifiable {
    case sgd = "SGD"
    case momentum = "Momentum"
    case nesterov = "Nesterov"
    case rmsProp = "RMSProp"
    case adaGrad = "AdaGrad"
    case adam = "Adam"

    var id: Self { self }

    var subtitle: String {
        switch self {
        case .sgd:
            return "Plain gradient descent"
        case .momentum:
            return "Velocity-smoothed updates"
        case .nesterov:
            return "Look-ahead momentum updates"
        case .rmsProp:
            return "Adaptive squared-gradient scaling"
        case .adaGrad:
            return "Cumulative per-parameter scaling"
        case .adam:
            return "Adaptive first and second moments"
        }
    }
}

enum LossKind: String, CaseIterable, Identifiable {
    case mse = "MSE"
    case mae = "MAE"
    case huber = "Huber"
    case logCosh = "Log-Cosh"
    case pseudoHuber = "Pseudo-Huber"
    case cauchy = "Cauchy"

    var id: Self { self }

    func value(prediction: Double, target: Double) -> Double {
        let diff = prediction - target
        switch self {
        case .mse:
            return diff * diff
        case .mae:
            return abs(diff)
        case .huber:
            let delta = 0.1
            if abs(diff) <= delta {
                return 0.5 * diff * diff
            }
            return delta * (abs(diff) - 0.5 * delta)
        case .logCosh:
            let magnitude = abs(diff)
            return magnitude + log1p(exp(-2.0 * magnitude)) - log(2.0)
        case .pseudoHuber:
            let delta = 0.1
            let scaled = diff / delta
            return delta * delta * (sqrt(1.0 + scaled * scaled) - 1.0)
        case .cauchy:
            let scale = 0.5
            let scaled = diff / scale
            return log1p(scaled * scaled)
        }
    }

    func derivative(prediction: Double, target: Double) -> Double {
        let diff = prediction - target
        switch self {
        case .mse:
            return 2.0 * diff
        case .mae:
            if diff == 0 {
                return 0
            }
            return diff.sign == .minus ? -1 : 1
        case .huber:
            let delta = 0.1
            if abs(diff) <= delta {
                return diff
            }
            return delta * (diff.sign == .minus ? -1 : 1)
        case .logCosh:
            return tanh(diff)
        case .pseudoHuber:
            let delta = 0.1
            let scaled = diff / delta
            return diff / sqrt(1.0 + scaled * scaled)
        case .cauchy:
            let scale = 0.5
            return (2.0 * diff) / ((scale * scale) + (diff * diff))
        }
    }
}

enum ActivationKind: String, CaseIterable, Identifiable {
    case sigmoid = "Sigmoid"
    case tanh = "Tanh"
    case relu = "ReLU"

    var id: Self { self }

    func activate(_ value: Double) -> Double {
        switch self {
        case .sigmoid:
            return 1.0 / (1.0 + exp(-value))
        case .tanh:
            return Foundation.tanh(value)
        case .relu:
            return max(0, value)
        }
    }

    func derivative(at preActivation: Double) -> Double {
        switch self {
        case .sigmoid:
            let sigmoid = activate(preActivation)
            return sigmoid * (1 - sigmoid)
        case .tanh:
            let value = Foundation.tanh(preActivation)
            return 1 - value * value
        case .relu:
            return preActivation > 0 ? 1 : 0
        }
    }
}

struct TrainingConfig: Equatable {
    static let sampleCountRange = 1...256
    static let epochRange = 200...2500

    var target: TargetKind = .reciprocalBell
    var sampleCount: Int = 96
    var seed: UInt64 = 42
    var optimizer: OptimizerKind = .momentum
    var loss: LossKind = .mse
    var activation: ActivationKind = .sigmoid
    var learningRate: Double = 0.08
    var epochCount: Int = 1500
    var noise: Double = 0.055
}
