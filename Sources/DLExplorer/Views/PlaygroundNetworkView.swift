import SwiftUI

struct PlaygroundNetworkView: View {
    let snapshot: NetworkSnapshot?
    let config: TrainingConfig
    let architecture: String

    var body: some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 16) {
                header
                summaryChips

                if let snapshot {
                    PlaygroundDiagramView(snapshot: snapshot)
                        .frame(maxWidth: .infinity, minHeight: 270)
                } else {
                    ProgressView("Preparing network view...")
                        .frame(maxWidth: .infinity, minHeight: 270)
                }

                Text("Data -> selected features -> hidden layers -> output. Node colors show activations for a probe input, and line color/thickness show current weights.")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }
            .padding(18)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    private var header: some View {
        HStack(alignment: .top) {
            VStack(alignment: .leading, spacing: 4) {
                Text("Network Playground")
                    .font(.headline)

                Text("A TensorFlow Playground-style structural view of the actual model being trained.")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }

            Spacer()

            Text("Widths \(architecture)")
                .font(.system(.subheadline, design: .monospaced))
                .padding(.horizontal, 10)
                .padding(.vertical, 6)
                .background(.quaternary, in: Capsule())
        }
    }

    private var summaryChips: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 8) {
                summaryChip(title: "Data", value: "\(config.sampleCount) samples")
                summaryChip(title: "Features", value: config.features.map(\.rawValue).joined(separator: ", "))
                summaryChip(title: "Hidden", value: config.activeHiddenLayerSizes.map(String.init).joined(separator: " -> "))
                summaryChip(title: "Output", value: "y_hat")
            }
            .padding(.vertical, 2)
        }
    }

    private func summaryChip(title: String, value: String) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(title)
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)
            Text(value)
                .font(.caption)
                .lineLimit(1)
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 8)
        .background(.quaternary.opacity(0.75), in: RoundedRectangle(cornerRadius: 10, style: .continuous))
    }
}

private struct PlaygroundDiagramView: View {
    let snapshot: NetworkSnapshot

    var body: some View {
        GeometryReader { proxy in
            let layout = PlaygroundLayout(snapshot: snapshot, size: proxy.size)

            ZStack {
                RoundedRectangle(cornerRadius: 16, style: .continuous)
                    .fill(.quinary)

                Canvas { context, _ in
                    for connection in snapshot.connections {
                        guard
                            let from = layout.position(forLayer: connection.fromLayerIndex, node: connection.fromNodeIndex),
                            let to = layout.position(forLayer: connection.toLayerIndex, node: connection.toNodeIndex)
                        else {
                            continue
                        }

                        var path = Path()
                        path.move(to: from)
                        path.addLine(to: to)

                        let stroke = StrokeStyle(
                            lineWidth: max(1.0, min(5.0, abs(connection.weight) * 2.8)),
                            lineCap: .round
                        )

                        context.stroke(
                            path,
                            with: .color(connection.weight >= 0 ? Color.blue.opacity(0.45) : Color.orange.opacity(0.45)),
                            style: stroke
                        )
                    }
                }

                ForEach(Array(snapshot.layers.enumerated()), id: \.offset) { layerIndex, layer in
                    Text(layer.title)
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(.secondary)
                        .position(x: layout.columnX(for: layerIndex), y: 18)

                    ForEach(Array(layer.nodes.enumerated()), id: \.element.id) { nodeIndex, node in
                        if let position = layout.position(forLayer: layerIndex, node: nodeIndex) {
                            PlaygroundNodeView(node: node)
                                .position(position)
                        }
                    }
                }

                VStack {
                    Spacer()
                    Text("Probe x = \(snapshot.probeX.formatted(.number.precision(.fractionLength(2))))")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .padding(.bottom, 8)
                }
            }
        }
    }
}

private struct PlaygroundNodeView: View {
    let node: NetworkNodeSnapshot

    var body: some View {
        VStack(spacing: 4) {
            ZStack {
                Circle()
                    .fill(nodeColor)
                    .overlay(
                        Circle()
                            .stroke(Color.primary.opacity(0.12), lineWidth: 1)
                    )
                    .frame(width: node.kind == .output ? 32 : 28, height: node.kind == .output ? 32 : 28)

                if node.kind == .hidden {
                    Text(shortIndex)
                        .font(.system(size: 9, weight: .bold, design: .rounded))
                        .foregroundStyle(.white.opacity(0.9))
                }
            }

            if node.kind != .hidden {
                Text(node.label)
                    .font(.system(size: 10, weight: .medium, design: .rounded))
                    .foregroundStyle(.secondary)
            }
        }
    }

    private var shortIndex: String {
        String(node.label.split(separator: ".").last ?? "H")
    }

    private var nodeColor: Color {
        let clipped = max(-1.0, min(1.0, node.value))
        if clipped >= 0 {
            return Color.blue.opacity(0.35 + (0.45 * clipped))
        }
        return Color.orange.opacity(0.35 + (0.45 * abs(clipped)))
    }
}

private struct PlaygroundLayout {
    let snapshot: NetworkSnapshot
    let size: CGSize

    func columnX(for layerIndex: Int) -> CGFloat {
        let inset: CGFloat = 44
        let usableWidth = max(size.width - (inset * 2), 1)
        let denominator = max(snapshot.layers.count - 1, 1)
        return inset + (CGFloat(layerIndex) * usableWidth / CGFloat(denominator))
    }

    func position(forLayer layerIndex: Int, node nodeIndex: Int) -> CGPoint? {
        guard snapshot.layers.indices.contains(layerIndex) else {
            return nil
        }

        let layer = snapshot.layers[layerIndex]
        guard layer.nodes.indices.contains(nodeIndex) else {
            return nil
        }

        let topInset: CGFloat = 54
        let bottomInset: CGFloat = 26
        let usableHeight = max(size.height - topInset - bottomInset, 1)
        let count = layer.nodes.count
        let y: CGFloat
        if count == 1 {
            y = topInset + usableHeight * 0.5
        } else {
            y = topInset + (CGFloat(nodeIndex) * usableHeight / CGFloat(count - 1))
        }

        return CGPoint(x: columnX(for: layerIndex), y: y)
    }
}
