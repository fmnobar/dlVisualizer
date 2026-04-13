import Charts
import SwiftUI

struct LossChartView: View {
    let snapshot: EpochSnapshot?
    let isTraining: Bool

    var body: some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 12) {
                HStack {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Loss Curve")
                            .font(.headline)

                        Text("The marker tracks the currently rendered epoch.")
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                    }

                    Spacer()

                    Text(isTraining ? "Streaming" : "Complete")
                        .font(.subheadline.weight(.medium))
                        .padding(.horizontal, 10)
                        .padding(.vertical, 6)
                        .background(.quaternary, in: Capsule())
                }

                Chart {
                    if let snapshot {
                        ForEach(snapshot.lossHistory) { point in
                            LineMark(
                                x: .value("Epoch", point.epoch),
                                y: .value("Loss", point.loss)
                            )
                            .foregroundStyle(Color.primary.opacity(0.9))
                        }

                        if let current = snapshot.lossHistory.last {
                            PointMark(
                                x: .value("Epoch", current.epoch),
                                y: .value("Loss", current.loss)
                            )
                            .foregroundStyle(Color.red)
                            .symbolSize(42)
                        }
                    }
                }
                .chartXScale(domain: 0...(snapshot?.totalEpochs ?? 1))
                .chartYScale(domain: 0...maxLoss)
                .chartXAxisLabel("epoch")
                .chartYAxisLabel("loss")
                .frame(maxWidth: .infinity, minHeight: 170)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    private var maxLoss: Double {
        let fallback = 0.1
        guard let snapshot else {
            return fallback
        }

        let maxObserved = snapshot.lossHistory.map(\.loss).max() ?? fallback
        return max(maxObserved * 1.15, 0.03)
    }
}
