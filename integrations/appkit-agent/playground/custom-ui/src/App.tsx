import { ChatProvider } from "@databricks/appkit-agent/chat-ui";
import "@databricks/appkit-agent/chat-ui/simple/styles.css";

function CustomChatApplication() {
  return <div>hello</div>
}

export function App() {
  return <ChatProvider>
    <CustomChatApplication />
  </ChatProvider>
}
