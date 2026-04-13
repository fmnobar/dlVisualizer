import SwiftUI

struct ContentView: View {
    let controller: TrainingController

    var body: some View {
        HSplitView {
            ControlSidebarView(controller: controller)
                .frame(minWidth: 290, idealWidth: 320, maxWidth: 340)

            ScrollView {
                rightColumn
            }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .background(.background)
    }

    private var rightColumn: some View {
        VStack(alignment: .leading, spacing: 18) {
            header

            PredictionChartView(
                targetSamples: controller.targetSamples,
                snapshot: controller.latestSnapshot,
                config: controller.config,
                architecture: controller.architecture,
                isTraining: controller.isTraining
            )
            .frame(maxWidth: .infinity, minHeight: 470)

            LossChartView(snapshot: controller.latestSnapshot, isTraining: controller.isTraining)
                .frame(maxWidth: .infinity, minHeight: 260)
        }
        .padding(24)
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("DL Explorer")
                .font(.system(size: 30, weight: .semibold, design: .rounded))

            Text("Watch a tiny network learn \(controller.targetEquation) from noisy data in real time.")
                .font(.title3)
                .foregroundStyle(.secondary)

            HStack(spacing: 12) {
                Label(controller.isTraining ? "Training" : "Idle", systemImage: controller.isTraining ? "bolt.fill" : "pause.circle")
                    .font(.subheadline.weight(.medium))
                    .padding(.horizontal, 10)
                    .padding(.vertical, 6)
                    .background(.quaternary, in: Capsule())

                Text("Seed \(controller.config.seed)")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }
        }
    }
}
