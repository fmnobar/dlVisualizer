import Charts
import SwiftUI

struct PredictionChartView: View {
    let targetSamples: [SamplePoint]
    let snapshot: EpochSnapshot?
    let config: TrainingConfig
    let architecture: String
    let isTraining: Bool

    var body: some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 14) {
                header
                chart
                footer
            }
            .padding(18)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    private var header: some View {
        HStack(alignment: .top) {
            VStack(alignment: .leading, spacing: 4) {
                Text("Prediction")
                    .font(.headline)

                Text("A real in-process MLP updates the blue curve as training progresses.")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }

            Spacer()

            VStack(alignment: .trailing, spacing: 6) {
                Text("Epoch \(snapshot?.epoch ?? 0)/\(snapshot?.totalEpochs ?? config.epochCount)")
                    .font(.subheadline.weight(.medium))
                    .padding(.horizontal, 10)
                    .padding(.vertical, 6)
                    .background(.quaternary, in: Capsule())

                if let snapshot {
                    Text("Loss \(snapshot.currentLoss.formatted(.number.precision(.fractionLength(4))))")
                        .font(.system(.subheadline, design: .monospaced))
                        .foregroundStyle(.secondary)
                }
            }
        }
    }

    private var chart: some View {
        Chart {
            ForEach(targetSamples) { point in
                PointMark(
                    x: .value("x", point.x),
                    y: .value("y", point.y)
                )
                .foregroundStyle(Color.red.opacity(0.85))
                .symbolSize(26)
            }

            if let snapshot {
                ForEach(snapshot.predictionCurve) { point in
                    LineMark(
                        x: .value("x", point.x),
                        y: .value("y", point.y)
                    )
                    .foregroundStyle(Color.blue)
                    .lineStyle(StrokeStyle(lineWidth: 3, lineCap: .round, lineJoin: .round))
                }
            }
        }
        .chartXScale(domain: RegressionDataFactory.xRange)
        .chartYScale(domain: yDomain)
        .chartYAxis {
            AxisMarks(position: .leading)
        }
        .frame(maxWidth: .infinity, minHeight: 380)
    }

    private var footer: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Blue: network prediction | Red: noisy targets")
                .font(.footnote)
                .foregroundStyle(.secondary)

            Text("Target \(config.target.rawValue) | Samples \(config.sampleCount) | Widths \(architecture)")
                .font(.footnote)
                .foregroundStyle(.secondary)

            Text("Activation \(config.activation.rawValue) | Loss \(config.loss.rawValue) | Optimizer \(config.optimizer.rawValue)\(isTraining ? " | Live" : "")")
                .font(.footnote)
                .foregroundStyle(.secondary)
        }
    }

    private var yDomain: ClosedRange<Double> {
        let yValues = targetSamples.map(\.y) + (snapshot?.predictionCurve.map(\.y) ?? [])
        guard let minY = yValues.min(), let maxY = yValues.max() else {
            return -1.0...1.0
        }

        let span = max(maxY - minY, 0.25)
        let padding = span * 0.12
        return (minY - padding)...(maxY + padding)
    }
}
