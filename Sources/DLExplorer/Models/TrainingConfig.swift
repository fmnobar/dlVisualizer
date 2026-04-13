import Foundation

enum OptimizerKind: String, CaseIterable, Identifiable {
    case sgd = "SGD"
    case momentum = "Momentum"
    case adam = "Adam"

    var id: Self { self }

    var subtitle: String {
        switch self {
        case .sgd:
            return "Plain gradient descent"
        case .momentum:
            return "Velocity-smoothed updates"
        case .adam:
            return "Adaptive first and second moments"
        }
    }
}

enum LossKind: String, CaseIterable, Identifiable {
    case mse = "MSE"
    case mae = "MAE"
    case huber = "Huber"

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
    var optimizer: OptimizerKind = .momentum
    var loss: LossKind = .mse
    var activation: ActivationKind = .sigmoid
    var learningRate: Double = 0.08
    var epochCount: Int = 1500
    var noise: Double = 0.055
}
