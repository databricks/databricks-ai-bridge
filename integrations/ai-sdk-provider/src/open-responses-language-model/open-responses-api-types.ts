/**
 * OpenResponses API request and response types
 * Used for agent apps that implement the OpenResponses streaming format
 */

export interface OpenResponsesRequest {
  messages: Array<{
    role: 'user' | 'assistant' | 'system';
    content: string;
  }>;
  model?: string;
  temperature?: number;
  max_tokens?: number;
  stream?: boolean;
}

export interface OpenResponsesResponse {
  role: string;
  content: string;
  model?: string;
  usage?: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}
