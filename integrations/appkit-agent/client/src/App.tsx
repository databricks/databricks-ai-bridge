import { Routes, Route } from 'react-router-dom';
import { ChatProvider } from '@/contexts/ChatProvider';
import { Toaster } from 'sonner';
import RootLayout from '@/layouts/RootLayout';
import ChatLayout from '@/layouts/ChatLayout';
import NewChatPage from '@/pages/NewChatPage';
import ChatPage from '@/pages/ChatPage';

function App() {
  return (
    <ChatProvider>
      <Toaster position="top-center" />
      <Routes>
        <Route path="/" element={<RootLayout />}>
          <Route element={<ChatLayout />}>
            <Route index element={<NewChatPage />} />
            <Route path="chat/:id" element={<ChatPage />} />
          </Route>
        </Route>
      </Routes>
    </ChatProvider>
  );
}

export default App;
