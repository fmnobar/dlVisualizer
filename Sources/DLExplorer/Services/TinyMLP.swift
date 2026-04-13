import Foundation

struct TinyMLP {
    struct DenseLayer {
        let inputSize: Int
        let outputSize: Int
        var weights: [Double]
        var biases: [Double]

        init(inputSize: Int, outputSize: Int, activation: ActivationKind?, generator: inout SeededRandomNumberGenerator) {
            self.inputSize = inputSize
            self.outputSize = outputSize

            let scale: Double
            if activation == .relu {
                scale = sqrt(2.0 / Double(inputSize))
            } else {
                scale = sqrt(1.0 / Double(inputSize))
            }

            self.weights = (0..<(inputSize * outputSize)).map { _ in
                generator.nextUniform(in: -scale...scale)
            }
            self.biases = Array(repeating: 0, count: outputSize)
        }

        func preactivate(_ input: [Double]) -> [Double] {
            var output = Array(repeating: 0.0, count: outputSize)

            for outputIndex in 0..<outputSize {
                var value = biases[outputIndex]
                let weightOffset = outputIndex * inputSize
                for inputIndex in 0..<inputSize {
                    value += weights[weightOffset + inputIndex] * input[inputIndex]
                }
                output[outputIndex] = value
            }

            return output
        }
    }

    struct ForwardCache {
        let activations: [[Double]]
        let preActivations: [[Double]]
    }

    struct NetworkGradients {
        var weightGrads: [[Double]]
        var biasGrads: [[Double]]

        static func zeros(for layers: [DenseLayer]) -> NetworkGradients {
            NetworkGradients(
                weightGrads: layers.map { Array(repeating: 0.0, count: $0.weights.count) },
                biasGrads: layers.map { Array(repeating: 0.0, count: $0.biases.count) }
            )
        }

        mutating func accumulate(layer layerIndex: Int, input: [Double], delta: [Double], layer: DenseLayer) {
            for outputIndex in 0..<layer.outputSize {
                let weightOffset = outputIndex * layer.inputSize
                for inputIndex in 0..<layer.inputSize {
                    weightGrads[layerIndex][weightOffset + inputIndex] += delta[outputIndex] * input[inputIndex]
                }
                biasGrads[layerIndex][outputIndex] += delta[outputIndex]
            }
        }

        mutating func scale(by factor: Double) {
            for layerIndex in weightGrads.indices {
                for weightIndex in weightGrads[layerIndex].indices {
                    weightGrads[layerIndex][weightIndex] *= factor
                }
                for biasIndex in biasGrads[layerIndex].indices {
                    biasGrads[layerIndex][biasIndex] *= factor
                }
            }
        }
    }

    struct OptimizerState {
        var firstMomentWeights: [[Double]]
        var firstMomentBiases: [[Double]]
        var secondMomentWeights: [[Double]]
        var secondMomentBiases: [[Double]]
        var step: Int = 0

        static func make(for layers: [DenseLayer]) -> OptimizerState {
            OptimizerState(
                firstMomentWeights: layers.map { Array(repeating: 0.0, count: $0.weights.count) },
                firstMomentBiases: layers.map { Array(repeating: 0.0, count: $0.biases.count) },
                secondMomentWeights: layers.map { Array(repeating: 0.0, count: $0.weights.count) },
                secondMomentBiases: layers.map { Array(repeating: 0.0, count: $0.biases.count) }
            )
        }
    }

    let activation: ActivationKind
    var layers: [DenseLayer]

    init(seed: UInt64, activation: ActivationKind, widths: [Int]) {
        self.activation = activation
        var generator = SeededRandomNumberGenerator(seed: seed)
        var builtLayers: [DenseLayer] = []

        for index in 0..<(widths.count - 1) {
            let layerActivation: ActivationKind? = index == widths.count - 2 ? nil : activation
            builtLayers.append(
                DenseLayer(
                    inputSize: widths[index],
                    outputSize: widths[index + 1],
                    activation: layerActivation,
                    generator: &generator
                )
            )
        }

        self.layers = builtLayers
    }

    func predict(inputs: [Double]) -> Double {
        forward(inputs).activations.last?.first ?? 0
    }

    func predict(x: Double, features: [FeatureKind]) -> Double {
        predict(inputs: RegressionDataFactory.featureVector(for: x, features: features))
    }

    func predictionCurve(xs: [Double], features: [FeatureKind]) -> [SamplePoint] {
        xs.map { x in
            SamplePoint(x: x, y: predict(x: x, features: features))
        }
    }

    func averageLoss(on examples: [TrainingExample], kind: LossKind) -> Double {
        guard !examples.isEmpty else {
            return 0
        }

        let total = examples.reduce(into: 0.0) { partialResult, example in
            partialResult += kind.value(prediction: predict(inputs: example.inputs), target: example.point.y)
        }
        return total / Double(examples.count)
    }

    func makeNetworkSnapshot(features: [FeatureKind], hiddenLayerSizes: [Int], probeX: Double) -> NetworkSnapshot {
        let inputValues = RegressionDataFactory.featureVector(for: probeX, features: features)
        let cache = forward(inputValues)

        var layersSnapshot: [NetworkLayerSnapshot] = [
            NetworkLayerSnapshot(
                id: "features",
                title: "Features",
                nodes: zip(features, inputValues).enumerated().map { index, pair in
                    let (feature, value) = pair
                    return NetworkNodeSnapshot(
                        id: "feature-\(index)",
                        label: feature.rawValue,
                        value: value,
                        kind: .feature
                    )
                }
            )
        ]

        for (layerIndex, width) in hiddenLayerSizes.enumerated() {
            let activations = cache.activations[layerIndex + 1]
            let nodes = (0..<width).map { neuronIndex in
                NetworkNodeSnapshot(
                    id: "hidden-\(layerIndex)-\(neuronIndex)",
                    label: "H\(layerIndex + 1).\(neuronIndex + 1)",
                    value: activations[neuronIndex],
                    kind: .hidden
                )
            }
            layersSnapshot.append(
                NetworkLayerSnapshot(
                    id: "hidden-\(layerIndex)",
                    title: "Hidden \(layerIndex + 1)",
                    nodes: nodes
                )
            )
        }

        let outputValue = cache.activations.last?.first ?? 0
        layersSnapshot.append(
            NetworkLayerSnapshot(
                id: "output",
                title: "Output",
                nodes: [
                    NetworkNodeSnapshot(
                        id: "output-0",
                        label: "y_hat",
                        value: outputValue,
                        kind: .output
                    )
                ]
            )
        )

        var connections: [NetworkConnectionSnapshot] = []
        for layerIndex in self.layers.indices {
            let denseLayer = self.layers[layerIndex]
            for outputIndex in 0..<denseLayer.outputSize {
                for inputIndex in 0..<denseLayer.inputSize {
                    let weightIndex = outputIndex * denseLayer.inputSize + inputIndex
                    connections.append(
                        NetworkConnectionSnapshot(
                            id: "connection-\(layerIndex)-\(inputIndex)-\(outputIndex)",
                            fromLayerIndex: layerIndex,
                            fromNodeIndex: inputIndex,
                            toLayerIndex: layerIndex + 1,
                            toNodeIndex: outputIndex,
                            weight: denseLayer.weights[weightIndex]
                        )
                    )
                }
            }
        }

        return NetworkSnapshot(
            probeX: probeX,
            layers: layersSnapshot,
            connections: connections
        )
    }

    mutating func trainEpoch(examples: [TrainingExample], config: TrainingConfig, optimizerState: inout OptimizerState) -> Double {
        guard !examples.isEmpty else {
            return 0
        }

        var gradients = NetworkGradients.zeros(for: layers)
        var epochLoss = 0.0
        let lastLayerIndex = layers.count - 1

        for example in examples {
            let cache = forward(example.inputs)
            let prediction = cache.activations[lastLayerIndex + 1][0]
            epochLoss += config.loss.value(prediction: prediction, target: example.point.y)

            var downstream = [config.loss.derivative(prediction: prediction, target: example.point.y)]
            gradients.accumulate(
                layer: lastLayerIndex,
                input: cache.activations[lastLayerIndex],
                delta: downstream,
                layer: layers[lastLayerIndex]
            )

            if lastLayerIndex > 0 {
                for layerIndex in stride(from: lastLayerIndex - 1, through: 0, by: -1) {
                    let nextLayer = layers[layerIndex + 1]
                    let currentLayer = layers[layerIndex]
                    var propagated = Array(repeating: 0.0, count: currentLayer.outputSize)

                    for neuronIndex in 0..<currentLayer.outputSize {
                        var sum = 0.0
                        for nextIndex in 0..<nextLayer.outputSize {
                            let weightIndex = nextIndex * nextLayer.inputSize + neuronIndex
                            sum += nextLayer.weights[weightIndex] * downstream[nextIndex]
                        }
                        propagated[neuronIndex] = sum * activation.derivative(at: cache.preActivations[layerIndex][neuronIndex])
                    }

                    downstream = propagated
                    gradients.accumulate(
                        layer: layerIndex,
                        input: cache.activations[layerIndex],
                        delta: downstream,
                        layer: currentLayer
                    )
                }
            }
        }

        let sampleScale = 1.0 / Double(examples.count)
        gradients.scale(by: sampleScale)
        epochLoss *= sampleScale

        apply(gradients: gradients, optimizer: config.optimizer, learningRate: config.learningRate, state: &optimizerState)
        return epochLoss
    }

    private func forward(_ input: [Double]) -> ForwardCache {
        var activations: [[Double]] = [input]
        var preActivations: [[Double]] = []

        for layerIndex in layers.indices {
            let preActivation = layers[layerIndex].preactivate(activations[layerIndex])
            preActivations.append(preActivation)

            let output: [Double]
            if layerIndex == layers.count - 1 {
                output = preActivation
            } else {
                output = preActivation.map(activation.activate)
            }

            activations.append(output)
        }

        return ForwardCache(activations: activations, preActivations: preActivations)
    }

    private mutating func apply(
        gradients: NetworkGradients,
        optimizer: OptimizerKind,
        learningRate: Double,
        state: inout OptimizerState
    ) {
        switch optimizer {
        case .sgd:
            for layerIndex in layers.indices {
                for weightIndex in layers[layerIndex].weights.indices {
                    layers[layerIndex].weights[weightIndex] -= learningRate * gradients.weightGrads[layerIndex][weightIndex]
                }
                for biasIndex in layers[layerIndex].biases.indices {
                    layers[layerIndex].biases[biasIndex] -= learningRate * gradients.biasGrads[layerIndex][biasIndex]
                }
            }

        case .momentum:
            let momentum = 0.9
            for layerIndex in layers.indices {
                for weightIndex in layers[layerIndex].weights.indices {
                    state.firstMomentWeights[layerIndex][weightIndex] =
                        momentum * state.firstMomentWeights[layerIndex][weightIndex]
                        - learningRate * gradients.weightGrads[layerIndex][weightIndex]
                    layers[layerIndex].weights[weightIndex] += state.firstMomentWeights[layerIndex][weightIndex]
                }
                for biasIndex in layers[layerIndex].biases.indices {
                    state.firstMomentBiases[layerIndex][biasIndex] =
                        momentum * state.firstMomentBiases[layerIndex][biasIndex]
                        - learningRate * gradients.biasGrads[layerIndex][biasIndex]
                    layers[layerIndex].biases[biasIndex] += state.firstMomentBiases[layerIndex][biasIndex]
                }
            }

        case .nesterov:
            let momentum = 0.9
            for layerIndex in layers.indices {
                for weightIndex in layers[layerIndex].weights.indices {
                    let previousVelocity = state.firstMomentWeights[layerIndex][weightIndex]
                    let velocity = momentum * previousVelocity - learningRate * gradients.weightGrads[layerIndex][weightIndex]
                    state.firstMomentWeights[layerIndex][weightIndex] = velocity
                    layers[layerIndex].weights[weightIndex] += (-momentum * previousVelocity) + ((1.0 + momentum) * velocity)
                }
                for biasIndex in layers[layerIndex].biases.indices {
                    let previousVelocity = state.firstMomentBiases[layerIndex][biasIndex]
                    let velocity = momentum * previousVelocity - learningRate * gradients.biasGrads[layerIndex][biasIndex]
                    state.firstMomentBiases[layerIndex][biasIndex] = velocity
                    layers[layerIndex].biases[biasIndex] += (-momentum * previousVelocity) + ((1.0 + momentum) * velocity)
                }
            }

        case .rmsProp:
            let decay = 0.99
            let epsilon = 1e-8
            for layerIndex in layers.indices {
                for weightIndex in layers[layerIndex].weights.indices {
                    let grad = gradients.weightGrads[layerIndex][weightIndex]
                    state.secondMomentWeights[layerIndex][weightIndex] =
                        decay * state.secondMomentWeights[layerIndex][weightIndex] + (1.0 - decay) * grad * grad
                    layers[layerIndex].weights[weightIndex] -= learningRate * grad / (sqrt(state.secondMomentWeights[layerIndex][weightIndex]) + epsilon)
                }
                for biasIndex in layers[layerIndex].biases.indices {
                    let grad = gradients.biasGrads[layerIndex][biasIndex]
                    state.secondMomentBiases[layerIndex][biasIndex] =
                        decay * state.secondMomentBiases[layerIndex][biasIndex] + (1.0 - decay) * grad * grad
                    layers[layerIndex].biases[biasIndex] -= learningRate * grad / (sqrt(state.secondMomentBiases[layerIndex][biasIndex]) + epsilon)
                }
            }

        case .adaGrad:
            let epsilon = 1e-8
            for layerIndex in layers.indices {
                for weightIndex in layers[layerIndex].weights.indices {
                    let grad = gradients.weightGrads[layerIndex][weightIndex]
                    state.secondMomentWeights[layerIndex][weightIndex] += grad * grad
                    layers[layerIndex].weights[weightIndex] -= learningRate * grad / (sqrt(state.secondMomentWeights[layerIndex][weightIndex]) + epsilon)
                }
                for biasIndex in layers[layerIndex].biases.indices {
                    let grad = gradients.biasGrads[layerIndex][biasIndex]
                    state.secondMomentBiases[layerIndex][biasIndex] += grad * grad
                    layers[layerIndex].biases[biasIndex] -= learningRate * grad / (sqrt(state.secondMomentBiases[layerIndex][biasIndex]) + epsilon)
                }
            }

        case .adam:
            state.step += 1
            let beta1 = 0.9
            let beta2 = 0.999
            let epsilon = 1e-8
            let correction1 = 1.0 - pow(beta1, Double(state.step))
            let correction2 = 1.0 - pow(beta2, Double(state.step))

            for layerIndex in layers.indices {
                for weightIndex in layers[layerIndex].weights.indices {
                    let grad = gradients.weightGrads[layerIndex][weightIndex]
                    state.firstMomentWeights[layerIndex][weightIndex] =
                        beta1 * state.firstMomentWeights[layerIndex][weightIndex] + (1.0 - beta1) * grad
                    state.secondMomentWeights[layerIndex][weightIndex] =
                        beta2 * state.secondMomentWeights[layerIndex][weightIndex] + (1.0 - beta2) * grad * grad

                    let mean = state.firstMomentWeights[layerIndex][weightIndex] / correction1
                    let variance = state.secondMomentWeights[layerIndex][weightIndex] / correction2
                    layers[layerIndex].weights[weightIndex] -= learningRate * mean / (sqrt(variance) + epsilon)
                }

                for biasIndex in layers[layerIndex].biases.indices {
                    let grad = gradients.biasGrads[layerIndex][biasIndex]
                    state.firstMomentBiases[layerIndex][biasIndex] =
                        beta1 * state.firstMomentBiases[layerIndex][biasIndex] + (1.0 - beta1) * grad
                    state.secondMomentBiases[layerIndex][biasIndex] =
                        beta2 * state.secondMomentBiases[layerIndex][biasIndex] + (1.0 - beta2) * grad * grad

                    let mean = state.firstMomentBiases[layerIndex][biasIndex] / correction1
                    let variance = state.secondMomentBiases[layerIndex][biasIndex] / correction2
                    layers[layerIndex].biases[biasIndex] -= learningRate * mean / (sqrt(variance) + epsilon)
                }
            }
        }
    }
}
