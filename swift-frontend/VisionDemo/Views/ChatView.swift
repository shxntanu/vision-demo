import LiveKit
import LiveKitComponents
import SwiftUI

struct ChatView: View {
    @EnvironmentObject var chatContext: ChatContext
    @EnvironmentObject var room: Room

    var body: some View {
        ZStack(alignment: room.localParticipant.isCameraEnabled() ? .top : .center) {
            VStack(alignment: .center) {
                Spacer()

                CameraView()
                    .frame(
                        width: CGFloat(chatContext.cameraDimensions.width) / UIScreen.main.scale,
                        height: CGFloat(chatContext.cameraDimensions.height) / UIScreen.main.scale
                    )
                    .opacity(room.localParticipant.isCameraEnabled() ? 1 : 0)
                    .animation(.snappy, value: room.localParticipant.isCameraEnabled())

                ActionBarView()
            }

            AgentView()
                .frame(
                    width: room.localParticipant.isCameraEnabled() ? 128 : 256,
                    height: room.localParticipant.isCameraEnabled() ? 128 : 256)
                .animation(.snappy, value: room.localParticipant.isCameraEnabled())
        }
    }
}
