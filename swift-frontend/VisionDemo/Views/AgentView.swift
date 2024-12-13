import LiveKit
import LiveKitComponents
import SwiftUI

struct AgentView: View {
    @EnvironmentObject var chatContext: ChatContext

    var body: some View {
        if let agent = chatContext.agentParticipant {
            AgentVisualizer(agent: agent)
        } else {
            AgentLoadingView()
        }
    }
}

struct AgentVisualizer: View {
    @ObservedObject var agent: RemoteParticipant

    var body: some View {
        if let track = agent.firstAudioTrack {
            BarAudioVisualizer(audioTrack: track, barColor: .contentPrimary, barCount: 5)
        } else {
            AgentLoadingView()
        }
    }
}

struct AgentLoadingView: View {
    var body: some View {
        Text("Waiting for agent…")
            .font(.subheadline)
            .foregroundStyle(.secondary)
    }
}
