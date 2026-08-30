import ReactDOM from 'react-dom/client';
import { getConfig } from '@/lib/config';
import { ChatApp } from '@/components/ChatApp';
import './index.css';

const rootElement = document.getElementById('root');
if (!rootElement) {
  throw new Error('Failed to find the root element with ID "root"');
}

const { basePath, apiBase } = getConfig();

ReactDOM.createRoot(rootElement).render(
  <ChatApp apiBase={apiBase} basePath={basePath} />,
);
