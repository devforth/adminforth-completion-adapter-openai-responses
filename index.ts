import type { AdapterOptions } from "./types.js";
import type { CompletionAdapter, CompletionStreamEvent, CompletionTool } from "adminforth";
import { encoding_for_model, type TiktokenModel } from "tiktoken";
import type OpenAI from "openai";

export type { AdapterOptions } from "./types.js";

type StreamChunkCallback = (
  chunk: string,
  event?: CompletionStreamEvent,
) => void | Promise<void>;

type ResponseCreateBody = OpenAI.Responses.ResponseCreateParams;
type OpenAIResponsesSuccess = OpenAI.Responses.Response;
type OpenAIErrorResponse = {
  error?: {
    message?: string;
    type?: string;
    param?: string | null;
    code?: string | null;
  };
};
type OpenAITool = OpenAI.Responses.Tool;
type OpenAIFunctionCall = Extract<
  OpenAI.Responses.ResponseOutputItem,
  { type: "function_call" }
>;

function extractOutputText(data: OpenAIResponsesSuccess): string {
  let text = "";

  for (const item of data.output ?? []) {
    if (item.type !== "message" || !Array.isArray(item.content)) continue;
    for (const part of item.content) {
      if (part.type === "output_text" && typeof part.text === "string") {
        text += part.text;
      }
    }
  }

  return text;
}

function extractReasoning(data: OpenAIResponsesSuccess): string | undefined {
  let reasoning = "";

  for (const item of data.output ?? []) {
    if (item.type !== "reasoning") continue;

    for (const part of item.summary ?? []) {
      if (part?.type === "summary_text" && typeof part.text === "string") {
        reasoning += part.text;
      }
    }

    if (!reasoning) {
      for (const part of item.content ?? []) {
        if (part?.type === "reasoning_text" && typeof part.text === "string") {
          reasoning += part.text;
        }
      }
    }
  }

  return reasoning || undefined;
}

function extractFunctionCall(
  data: OpenAIResponsesSuccess,
): OpenAIFunctionCall | undefined {
  for (const item of data.output ?? []) {
    if (item.type === "function_call") {
      return item;
    }
  }

  return undefined;
}

async function executeToolCall(
  toolCall: OpenAIFunctionCall,
  tools?: CompletionTool[],
): Promise<string> {
  const tool = tools?.find((candidate) => candidate.name === toolCall.name);
  if (!tool) {
    throw new Error(`Tool "${toolCall.name}" not found`);
  }

  const toolResult = await tool.handler(JSON.parse(toolCall.arguments));
  if (typeof toolResult === "string") return toolResult;
  if (typeof toolResult === "undefined") return "";
  return JSON.stringify(toolResult);
}

function parseSseBlock(block: string) {
  let event: string | undefined;
  let data = "";

  for (const rawLine of block.split("\n")) {
    const line = rawLine.trimEnd();
    if (!line) continue;
    if (line.startsWith("event:")) event = line.slice(6).trim();
    if (line.startsWith("data:")) data += line.slice(5).trim();
  }

  return data ? { event, data } : null;
}

export default class CompletionAdapterOpenAIResponses
  implements CompletionAdapter
{
  options: AdapterOptions;
  private encoding: ReturnType<typeof encoding_for_model>;

  constructor(options: AdapterOptions) {
    this.options = options;
    try {
      this.encoding = encoding_for_model(
        (this.options.model || "gpt-5-nano") as TiktokenModel,
      );
    } catch (error) {
      console.warn(
        `Failed to initialize tiktoken tokenizer for model "${this.options.model}", falling back to "gpt-5-nano". Error:`,
      );
      this.encoding = encoding_for_model("gpt-5-nano" as TiktokenModel);
    }
  }

  validate() {
    if (!this.options.openAiApiKey) {
      throw new Error("openAiApiKey is required");
    }
  }

  measureTokensCount(content: string): number {
    return this.encoding.encode(content).length;
  }

  complete = async (
    content: string,
    maxTokens = 50,
    outputSchema?: any,
    reasoningEffort: "none" | "minimal" | "low" | "medium" | "high" | "xhigh" = "low",
    toolsOrOnChunk?: CompletionTool[] | StreamChunkCallback,
    onChunk?: StreamChunkCallback,
  ): Promise<{
    content?: string;
    finishReason?: string;
    error?: string;
  }> => {
    const model = this.options.model || "gpt-5-nano";
    const tools = Array.isArray(toolsOrOnChunk) ? toolsOrOnChunk : undefined;
    const streamChunkCallback =
      typeof toolsOrOnChunk === "function" ? toolsOrOnChunk : onChunk;
    const isStreaming = typeof streamChunkCallback === "function";
    const extra = this.options.extraRequestBodyParameters;
    let openAiTools: OpenAITool[] | undefined = undefined;
    if (tools && tools.length > 0) {
      openAiTools = tools.map((tool) => ({
        type: "function",
        name: tool.name,
        description: tool.description,
        parameters: tool.input_schema,
        strict: true,
      }));
    }

    const body = {
      model,
      input: content,
      max_output_tokens: maxTokens,
      stream: isStreaming,
      text: outputSchema
        ? {
            format: {
              type: "json_schema",
              ...outputSchema,
            },
          }
        : {
            format: {
              type: "text",
            },
          },
      reasoning: {
        effort: reasoningEffort,
        summary: "auto",
      },
      tools: openAiTools,
      ...extra,
    } as ResponseCreateBody;

    const resp = await fetch("https://api.openai.com/v1/responses", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${this.options.openAiApiKey}`,
      },
      body: JSON.stringify(body),
    });

    if (!resp.ok) {
      let errorMessage = `OpenAI request failed with status ${resp.status}`;
      try {
        const errorData = (await resp.json()) as OpenAIErrorResponse;
        if (errorData.error?.message) errorMessage = errorData.error.message;
      } catch {}
      return { error: errorMessage };
    }

    if (!isStreaming) {
      const json = await resp.json();
      const data = json as OpenAIResponsesSuccess & OpenAIErrorResponse;
      if (data.error) {
        return { error: data.error.message };
      }

      const toolCall = extractFunctionCall(data);
      if (toolCall) {
        try {
          const toolResult = await executeToolCall(toolCall, tools);
          return {
            content: toolResult,
            finishReason: "tool_call",
          };
        } catch (error: any) {
          return {
            error: error?.message || "Tool execution failed",
            finishReason: "tool_call",
          };
        }
      }

      const parsedContent = extractOutputText(data);

      return {
        content: parsedContent,
        finishReason: data.incomplete_details?.reason
          ? data.incomplete_details.reason
          : undefined,
      };
    }

    if (!resp.body) {
      return { error: "Response body is empty" };
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder("utf-8");

    let buffer = "";
    let fullContent = "";
    let fullReasoning = "";
    let finishReason: string | undefined;
    let completedResponse: OpenAIResponsesSuccess | undefined;

    const handleEvent = async (event: any, eventType?: string) => {
      const type = event?.type || eventType;

      if (type === "response.output_text.delta") {
        const delta = event?.delta || "";
        if (!delta) return;
        fullContent += delta;
        await streamChunkCallback?.(delta, { type: "output", delta, text: fullContent });
        return;
      }

      if (
        type === "response.reasoning_summary_text.delta" ||
        type === "response.reasoning_text.delta"
      ) {
        const delta = event?.delta || "";
        if (!delta) return;
        fullReasoning += delta;
        await streamChunkCallback?.(delta, {
          type: "reasoning",
          delta,
          text: fullReasoning,
        });
        return;
      }

      if (type === "response.completed" || type === "response.incomplete") {
        const response = event?.response as OpenAIResponsesSuccess | undefined;
        if (!response) return;

        const finalContent = extractOutputText(response);
        if (finalContent.startsWith(fullContent)) {
          const delta = finalContent.slice(fullContent.length);
          if (delta) {
            fullContent = finalContent;
            await streamChunkCallback?.(delta, {
              type: "output",
              delta,
              text: fullContent,
            });
          }
        }

        const finalReasoning = extractReasoning(response) || "";
        if (finalReasoning.startsWith(fullReasoning)) {
          const delta = finalReasoning.slice(fullReasoning.length);
          if (delta) {
            fullReasoning = finalReasoning;
            await streamChunkCallback?.(delta, {
              type: "reasoning",
              delta,
              text: fullReasoning,
            });
          }
        }

        finishReason =
          response.incomplete_details?.reason || response.status || finishReason;
        completedResponse = response;
        return;
      }

      if (type === "response.failed") {
        throw new Error(
          event?.response?.error?.message ||
            event?.error?.message ||
            "Response failed",
        );
      }
    };

    try {
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        const blocks = buffer.split("\n\n");
        buffer = blocks.pop() || "";

        for (const block of blocks) {
          const parsedBlock = parseSseBlock(block);
          if (!parsedBlock?.data || parsedBlock.data === "[DONE]") continue;

          let event: any;
          try {
            event = JSON.parse(parsedBlock.data);
          } catch {
            continue;
          }

          if (event?.error?.message) {
            return { error: event.error.message };
          }

          await handleEvent(event, parsedBlock.event);
        }
      }

      if (buffer.trim()) {
        const parsedBlock = parseSseBlock(buffer.trim());
        if (parsedBlock?.data && parsedBlock.data !== "[DONE]") {
          try {
            await handleEvent(JSON.parse(parsedBlock.data), parsedBlock.event);
          } catch (error: any) {
            return {
              error: error?.message || "Streaming failed",
              content: fullContent || undefined,
              finishReason,
            };
          }
        }
      }

      if (completedResponse) {
        const toolCall = extractFunctionCall(completedResponse);
        if (toolCall) {
          try {
            const toolResult = await executeToolCall(toolCall, tools);
            if (toolResult) {
              const delta = toolResult.startsWith(fullContent)
                ? toolResult.slice(fullContent.length)
                : toolResult;
              if (delta) {
                await streamChunkCallback?.(delta, {
                  type: "output",
                  delta,
                  text: toolResult,
                });
              }
            }

            return {
              content: toolResult,
              finishReason: "tool_call",
            };
          } catch (error: any) {
            return {
              error: error?.message || "Tool execution failed",
              content: fullContent || undefined,
              finishReason: "tool_call",
            };
          }
        }
      }

      return {
        content: fullContent || undefined,
        finishReason,
      };
    } catch (error: any) {
      return {
        error: error?.message || "Streaming failed",
        content: fullContent || undefined,
        finishReason,
      };
    } finally {
      reader.releaseLock();
    }
  };
}