/* Text shown in chat is intentionally narrower than the raw event inspector. */
(function (root) {
  "use strict";
  const textTypes = new Set(["text", "input_text", "output_text"]);

  function extractText(content) {
    if (typeof content === "string") return content;
    if (Array.isArray(content)) return content.map(extractText).filter(Boolean).join("\n");
    if (!content || typeof content !== "object") return "";
    if (textTypes.has(content.type) && typeof content.text === "string") return content.text;
    if (content.type === "refusal" && typeof content.refusal === "string") return content.refusal;
    // Reasoning, signatures, encrypted data and unknown typed blocks are not answer text.
    return "";
  }

  function reasoningSummary(content) {
    if (!Array.isArray(content)) return "";
    return content.flatMap((part) => part?.type === "reasoning" && Array.isArray(part.summary)
      ? part.summary.filter((item) => item?.type === "summary_text" && typeof item.text === "string").map((item) => item.text)
      : []).join("\n");
  }

  function errorText(error) {
    if (typeof error === "string") return error;
    if (Array.isArray(error)) return error.map(errorText).filter(Boolean).join("; ");
    if (error && typeof error === "object") {
      const message = error.message || error.msg || "Request failed";
      return error.code ? `${error.code}: ${message}` : message;
    }
    return "Request failed";
  }

  let markdown;
  function renderMarkdown(text) {
    // Locally vendored markdown-it: raw HTML disabled, unsafe link schemes rejected by the
    // parser, and remote images disabled so a model response cannot trigger background requests.
    markdown ||= root.markdownit({ html: false, linkify: false, breaks: true }).disable("image");
    return markdown.render(text);
  }

  const api = { extractText, reasoningSummary, errorText, renderMarkdown };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.ChatContent = api;
})(globalThis);
