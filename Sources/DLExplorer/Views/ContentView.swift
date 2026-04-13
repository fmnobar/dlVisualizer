import SwiftUI

struct ContentView: View {
    let controller: TrainingController

    var body: some View {
        HSplitView {
            ControlSidebarView(controller: controller)
                .frame(minWidth: 290, idealWidth: 320, maxWidth: 340)

            rightColumn
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
            .frame(maxWidth: .infinity, minHeight: 430)

            LossChartView(snapshot: controller.latestSnapshot, isTraining: controller.isTraining)
                .frame(maxWidth: .infinity, minHeight: 210)
        }
        .padding(24)
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

                Text("Seed \(controller.seed)")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }
        }
    }
}
