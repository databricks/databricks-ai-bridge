import {
  generateText,
  type LanguageModelUsage,
  type UIMessageStreamWriter,
} from "ai";
import { generateUUID } from "./utils";

/**
 * Reads all chunks from a UI message stream, forwarding non-error parts to the
 * writer. Returns whether the stream encountered any errors.
 *
 * Pre-first-chunk errors trigger a fallback; mid-stream errors are forwarded.
 */
export async function drainStreamToWriter(
  uiStream: ReadableStream,
  writer: UIMessageStreamWriter,
): Promise<{ failed: boolean; errorText?: string }> {
  const reader = uiStream.getReader();
  let receivedTextChunk = false;

  try {
    for (
      let chunk = await reader.read();
      !chunk.done;
      chunk = await reader.read()
    ) {
      if (chunk.value.type === "error") {
        if (!receivedTextChunk) {
          return { failed: true, errorText: chunk.value.errorText };
        }
        writer.write(chunk.value);
      } else {
        if (!receivedTextChunk && chunk.value.type.startsWith("text-")) {
          receivedTextChunk = true;
        }
        writer.write(chunk.value);
      }
    }
  } catch (readError) {
    if (!receivedTextChunk) {
      return { failed: true };
    }
  } finally {
    reader.releaseLock();
  }

  return { failed: false };
}

/**
 * Converts a generateText result's content array into UIMessageChunks and
 * writes them to the stream writer. Mirrors the transform in the AI SDK's
 * streamText().toUIMessageStream().
 */
function writeGenerateTextResultToStream(
  result: Awaited<ReturnType<typeof generateText>>,
  writer: UIMessageStreamWriter,
) {
  for (const part of result.content) {
    const id = generateUUID();

    switch (part.type) {
      case "text": {
        if (part.text.length > 0) {
          writer.write({ type: "text-start", id });
          writer.write({ type: "text-delta", id, delta: part.text });
          writer.write({ type: "text-end", id });
        }
        break;
      }
      case "reasoning": {
        if (part.text.length > 0) {
          writer.write({ type: "reasoning-start" as "text-start", id });
          writer.write({
            type: "reasoning-delta" as "text-delta",
            id,
            delta: part.text,
          });
          writer.write({ type: "reasoning-end" as "text-end", id });
        }
        break;
      }
      case "file": {
        writer.write({
          type: "file",
          mediaType: part.file.mediaType,
          url: `data:${part.file.mediaType};base64,${part.file.base64}`,
        } as Parameters<typeof writer.write>[0]);
        break;
      }
      case "tool-call": {
        writer.write({
          type: "tool-input-available",
          toolCallId: part.toolCallId,
          toolName: part.toolName,
          input: part.input,
          dynamic: part.dynamic,
        } as Parameters<typeof writer.write>[0]);
        break;
      }
      case "tool-result": {
        writer.write({
          type: "tool-output-available",
          toolCallId: part.toolCallId,
          output: part.output,
        } as Parameters<typeof writer.write>[0]);
        break;
      }
      case "source": {
        if (part.sourceType === "url") {
          writer.write({
            type: "source-url",
            sourceId: part.id,
            url: part.url,
            title: part.title,
            ...(part.providerMetadata != null
              ? { providerMetadata: part.providerMetadata }
              : {}),
          } as Parameters<typeof writer.write>[0]);
        } else if (part.sourceType === "document") {
          writer.write({
            type: "source-document",
            sourceId: part.id,
            mediaType: part.mediaType,
            title: part.title,
            filename: part.filename,
            ...(part.providerMetadata != null
              ? { providerMetadata: part.providerMetadata }
              : {}),
          } as Parameters<typeof writer.write>[0]);
        }
        break;
      }
      default:
        break;
    }
  }

  writer.write({ type: "finish", finishReason: result.finishReason });
}

/**
 * Falls back to a non-streaming generateText call and writes the result as
 * stream parts. Returns usage on success, undefined on failure.
 */
export async function fallbackToGenerateText(
  params: Parameters<typeof generateText>[0],
  writer: UIMessageStreamWriter,
): Promise<{ usage: LanguageModelUsage; traceId?: string } | undefined> {
  try {
    const fallback = await generateText(params);

    const traceId = (
      fallback?.response?.body as {
        metadata?: { trace_id?: string };
      }
    )?.metadata?.trace_id;

    writeGenerateTextResultToStream(fallback, writer);
    return { usage: fallback.usage, traceId };
  } catch (fallbackError) {
    const errorMessage =
      fallbackError instanceof Error
        ? fallbackError.message
        : String(fallbackError);
    writer.write({ type: "data-error", data: errorMessage } as Parameters<
      typeof writer.write
    >[0]);
    return undefined;
  }
}
