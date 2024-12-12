import SwiftUI

struct ConnectionView: View {
    @EnvironmentObject private var chatContext: ChatContext

    @State private var isConnecting: Bool = false
    private var tokenService: TokenService = TokenService()

    var body: some View {
        if chatContext.isConnected {
            ChatView()
        } else {
            VStack {
                Button(action: {
                    Task {
                        isConnecting = true

                        let roomName = "room-\(Int.random(in: 1000...9999))"
                        let participantName = "user-\(Int.random(in: 1000...9999))"

                        do {
                            let connectionDetails = try await tokenService.fetchConnectionDetails(
                                roomName: roomName,
                                participantName: participantName
                            )

                            try await chatContext.connect(url: connectionDetails.serverUrl, token: connectionDetails.participantToken)
                        } catch {
                            print("Connection error: \(error)")
                        }
                        isConnecting = false
                    }
                }) {
                    Label("Connect",
                          systemImage: "waveform")
                    .fontWeight(.bold)
                }
                .buttonStyle(.circleButton(isActive: false))
                .disabled(isConnecting)
            }
        }
    }
}
