import SwiftUI

struct ControlSidebarView: View {
    let controller: TrainingController

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                titleBlock
                dataSection
                modelSection
                trainingSection
                actionsSection
            }
            .padding(18)
        }
        .background(.regularMaterial)
    }

    private var titleBlock: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Experiment Controls")
                .font(.title3.weight(.semibold))

            Text("Every change restarts training from fresh weights so the charts show the true impact.")
                .font(.subheadline)
                .foregroundStyle(.secondary)
        }
    }

    private var dataSection: some View {
        GroupBox("Data") {
            VStack(alignment: .leading, spacing: 14) {
                menuRow(
                    title: "Target",
                    selection: Binding(
                        get: { controller.config.target },
                        set: { value in
                            controller.updateConfig { $0.target = value }
                        }
                    ),
                    values: TargetKind.allCases
                )

                Text(controller.targetEquation)
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)

                editableIntegerRow(
                    title: "Samples",
                    value: Binding(
                        get: { controller.config.sampleCount },
                        set: { newValue in
                            controller.updateConfig { config in
                                config.sampleCount = newValue
                            }
                        }
                    )
                )

                editableIntegerRow(
                    title: "Seed",
                    value: Binding(
                        get: {
                            if controller.config.seed > UInt64(Int.max) {
                                return Int.max
                            }
                            return Int(controller.config.seed)
                        },
                        set: { newValue in
                            controller.updateConfig { config in
                                config.seed = UInt64(max(0, newValue))
                            }
                        }
                    )
                )

                VStack(alignment: .leading, spacing: 6) {
                    HStack {
                        Text("Noise")
                        Spacer()
                        Text(String(format: "%.3f", controller.config.noise))
                            .monospacedDigit()
                            .foregroundStyle(.secondary)
                    }

                    Slider(
                        value: Binding(
                            get: { controller.config.noise },
                            set: { value in
                                controller.updateConfig { $0.noise = value }
                            }
                        ),
                        in: 0...0.16
                    )
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    private var modelSection: some View {
        GroupBox("Model") {
            VStack(alignment: .leading, spacing: 14) {
                labeledValue("Widths", value: controller.architecture)

                VStack(alignment: .leading, spacing: 6) {
                    Text("Activation")
                    Picker("Activation", selection: Binding(
                        get: { controller.config.activation },
                        set: { value in
                            controller.updateConfig { $0.activation = value }
                        }
                    )) {
                        ForEach(ActivationKind.allCases) { activation in
                            Text(activation.rawValue).tag(activation)
                        }
                    }
                    .labelsHidden()
                    .pickerStyle(.menu)
                    .frame(maxWidth: .infinity, alignment: .leading)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    private var trainingSection: some View {
        GroupBox("Training") {
            VStack(alignment: .leading, spacing: 14) {
                menuRow(
                    title: "Optimizer",
                    selection: Binding(
                        get: { controller.config.optimizer },
                        set: { value in
                            controller.updateConfig { $0.optimizer = value }
                        }
                    ),
                    values: OptimizerKind.allCases
                )

                menuRow(
                    title: "Loss",
                    selection: Binding(
                        get: { controller.config.loss },
                        set: { value in
                            controller.updateConfig { $0.loss = value }
                        }
                    ),
                    values: LossKind.allCases
                )

                VStack(alignment: .leading, spacing: 6) {
                    HStack {
                        Text("Learning Rate")
                        Spacer()
                        Text(String(format: "%.3f", controller.config.learningRate))
                            .monospacedDigit()
                            .foregroundStyle(.secondary)
                    }

                    Slider(
                        value: Binding(
                            get: { controller.config.learningRate },
                            set: { value in
                                controller.updateConfig { $0.learningRate = value }
                            }
                        ),
                        in: 0.001...0.15
                    )
                }

                Stepper(
                    value: Binding(
                        get: { controller.config.epochCount },
                        set: { value in
                            controller.updateConfig { $0.epochCount = value }
                        }
                    ),
                    in: 200...2500,
                    step: 100
                ) {
                    HStack {
                        Text("Epochs")
                        Spacer()
                        Text("\(controller.config.epochCount)")
                            .monospacedDigit()
                            .foregroundStyle(.secondary)
                    }
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    private var actionsSection: some View {
        GroupBox("Actions") {
            VStack(alignment: .leading, spacing: 10) {
                Button("Rerun Training") {
                    controller.rerun()
                }
                .buttonStyle(.borderedProminent)

                Button("Reset Seed") {
                    controller.resetSeed()
                }
                .buttonStyle(.bordered)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    private func labeledValue(_ title: String, value: String) -> some View {
        HStack(alignment: .firstTextBaseline) {
            Text(title)
            Spacer(minLength: 12)
            Text(value)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.trailing)
        }
        .font(.subheadline)
    }

    private func editableIntegerRow(title: String, value: Binding<Int>) -> some View {
        HStack(alignment: .firstTextBaseline) {
            Text(title)
            Spacer(minLength: 12)
            TextField(title, value: value, format: .number)
                .textFieldStyle(.roundedBorder)
                .multilineTextAlignment(.trailing)
                .frame(width: 88)
        }
        .font(.subheadline)
    }

    private func menuRow<Value: Hashable & RawRepresentable & Identifiable>(
        title: String,
        selection: Binding<Value>,
        values: [Value]
    ) -> some View where Value.RawValue == String {
        VStack(alignment: .leading, spacing: 6) {
            Text(title)
            Picker(title, selection: selection) {
                ForEach(values) { value in
                    Text(value.rawValue).tag(value)
                }
            }
            .labelsHidden()
            .pickerStyle(.menu)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }
}
