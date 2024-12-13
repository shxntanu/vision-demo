import LiveKit
import LiveKitComponents
import SwiftUI

struct AgentView: View {
    @EnvironmentObject var chatContext: ChatContext
    @Environment(\.colorScheme) private var colorScheme
    @State private var isPulsing = false

    var body: some View {
        ZStack {
            Circle()
                .fill(colorScheme == .dark ? Color.blue.opacity(0.4) : Color.blue.opacity(0.2))
                .frame(width: 100, height: 100)
                .scaleEffect(isPulsing ? 1.1 : 1.0)
                .opacity(isPulsing ? 0.8 : 1.0)
                .animation(
                    Animation.easeInOut(duration: 1.0).repeatForever(autoreverses: true),
                    value: isPulsing
                )

            if let agent = chatContext.agentParticipant {
                AgentAudioVisualizer(agent: agent)
            }
        }.frame(width: 100, height: 100)
            .onAppear {
                updatePulsing()
            }
            .onChange(of: chatContext.agentParticipant) {
                updatePulsing()
            }
    }

    private func updatePulsing() {
        isPulsing = chatContext.agentParticipant == nil
    }
}

struct AgentAudioVisualizer: View {
    @Environment(\.colorScheme) private var colorScheme
    @ObservedObject var agent: RemoteParticipant

    private var opacity: Double {
        switch colorScheme {
        case .dark:
            return 0.8
        default:
            return 0.5
        }
    }

    var body: some View {
        ZStack {
            if let track = agent.firstAudioTrack {
                BarAudioVisualizer(audioTrack: track, barColor: .blue, barCount: 3, barMinOpacity: opacity)
                    .frame(width: 64, height: 64)
            } else {
                Circle().fill(.blue.opacity(opacity)).frame(width: 20, height: 20)
            }
        }
    }
}

#Preview {
    AgentView()
        .environmentObject(ChatContext())
}
