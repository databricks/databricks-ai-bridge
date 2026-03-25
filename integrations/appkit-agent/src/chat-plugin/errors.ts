export class ChatServerError extends Error {
  constructor(
    readonly code: string,
    message?: string,
  ) {
    super(message ?? code);
    this.name = "ChatServerError";
  }

  toResponse(): { status: number; json: { error: string; code: string } } {
    const statusByCode: Record<string, number> = {
      "bad_request:api": 400,
      "unauthorized:chat": 401,
      "forbidden:chat": 403,
      "not_found:database": 404,
      "empty:stream": 204,
      "offline:chat": 503,
      "bad_request:database": 500,
    };
    return {
      status: statusByCode[this.code] ?? 500,
      json: { error: this.message, code: this.code },
    };
  }
}
