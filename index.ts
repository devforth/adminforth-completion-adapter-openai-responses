import type { AdapterOptions } from "./types.js";
import type {
  CompletionAdapter,
  CompletionStreamEvent,
  CompletionTool,
} from "adminforth";
import { AIMessage } from "@langchain/core/messages";
import { ChatOpenAI } from "@langchain/openai";
import { createMiddleware } from "langchain";
import { encoding_for_model, type TiktokenModel } from "tiktoken";
import type OpenAI from "openai";

export type { AdapterOptions } from "./types.js";

type StreamChunkCallback = (
  chunk: string,
  event?: CompletionStreamEvent,
) => void | Promise<void>;

type ReasoningEffort =
  | "none"
  | "minimal"
  | "low"
  | "medium"
  | "high"
  | "xhigh";

type AgentModelPurpose = "primary" | "summary";

type CompletionRequestInput = {
  content: string;
  maxTokens?: number;
  outputSchema?: any;
  reasoningEffort?: ReasoningEffort;
  tools?: CompletionTool[];
  onChunk?: StreamChunkCallback;
};

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

type OpenAiResponsesMetadata = {
  id?: string;
};

type OpenAiResponsesContext = {
  sessionId: string;
  turnId: string;
};

const DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1";
const RAW_REQUEST_LOG_PREFIX = "[CompletionAdapterOpenAIResponses] Raw /responses request";

type FetchInput = Parameters<typeof fetch>[0];
type FetchInit = Parameters<typeof fetch>[1];

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

function getAgentReasoningEffort(
  purpose: AgentModelPurpose,
): Exclude<ReasoningEffort, "none"> {
  return purpose === "summary" ? "minimal" : "low";
}

function getTurnKey(context: OpenAiResponsesContext) {
  return `${context.sessionId}:${context.turnId}`;
}

function getResponseId(message: AIMessage) {
  const metadata = message.response_metadata as OpenAiResponsesMetadata | undefined;
  return metadata?.id ?? null;
}

function getPreviousResponseId(modelSettings?: Record<string, unknown>) {
  return (modelSettings as { previous_response_id?: string } | undefined)
    ?.previous_response_id;
}

function getContinuationMessages<T extends { response_metadata?: unknown }>(
  messages: T[],
  previousResponseId: string,
) {
  let continuationStartIndex: number | null = null;

  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index];

    if (
      AIMessage.isInstance(message) &&
      (message.response_metadata as OpenAiResponsesMetadata | undefined)?.id ===
        previousResponseId
    ) {
      continuationStartIndex = index + 1;
      break;
    }
  }

  if (continuationStartIndex === null) {
    return null;
  }

  return messages.slice(continuationStartIndex);
}

function createOpenAiResponsesContinuationMiddleware() {
  const responseIdsByTurn = new Map<string, string>();

  return createMiddleware({
    name: "OpenAiResponsesContinuationMiddleware",
    async wrapModelCall(request, handler) {
      const context = request.runtime.context as OpenAiResponsesContext;
      const turnKey = getTurnKey(context);
      const previousResponseId =
        getPreviousResponseId(request.modelSettings) ??
        responseIdsByTurn.get(turnKey);
      const continuationMessages = previousResponseId
        ? getContinuationMessages(request.messages, previousResponseId)
        : null;

      const response = (await handler(
        previousResponseId && continuationMessages
          ? {
              ...request,
              messages: continuationMessages,
              modelSettings: {
                ...request.modelSettings,
                previous_response_id: previousResponseId,
              },
            }
          : request,
      )) as AIMessage;

      const responseId = getResponseId(response);

      if (responseId) {
        responseIdsByTurn.set(turnKey, responseId);
      } else {
        responseIdsByTurn.delete(turnKey);
      }

      return response;
    },
  });
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
      // console.warn(
      //   `Failed to initialize tiktoken tokenizer for model "${this.options.model}", falling back to "gpt-5-nano". Error:`,
      // );
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

  private getConfiguredBaseUrl() {
    return this.options.baseUrl;
  }

  private shouldDumpRawRequest() {
    return this.options.dumpRawRequest === true;
  }

  private getClientConfiguration() {
    const configuredBaseUrl = this.getConfiguredBaseUrl();
    const debugFetch = this.shouldDumpRawRequest()
      ? this.createResponsesDebugFetch()
      : undefined;

    if (!configuredBaseUrl && !debugFetch) {
      return undefined;
    }

    return {
      ...(configuredBaseUrl ? { baseURL: configuredBaseUrl } : {}),
      ...(debugFetch ? { fetch: debugFetch } : {}),
    };
  }

  private createResponsesDebugFetch() {
    return async (input: FetchInput, init?: FetchInit) => {
      const url = this.getFetchUrl(input);

      if (this.isResponsesUrl(url) && typeof init?.body === "string") {
        this.dumpRawRequest(url, init.body);
      }

      return fetch(input, init);
    };
  }

  private getFetchUrl(input: FetchInput) {
    if (typeof input === "string") {
      return input;
    }

    if (input instanceof URL) {
      return input.toString();
    }

    return input.url;
  }

  private isResponsesUrl(url: string) {
    try {
      return new URL(url).pathname.endsWith("/responses");
    } catch {
      return url.endsWith("/responses") || url.includes("/responses?");
    }
  }

  private dumpRawRequest(url: string, body: string) {
    console.info(`${RAW_REQUEST_LOG_PREFIX} ${url}`);
    try {
      console.info(JSON.stringify(JSON.parse(body), null, 2));
    } catch {
      console.info(body);
    }
  }

  private getResponsesUrl() {
    const baseUrl = this.getConfiguredBaseUrl() || DEFAULT_OPENAI_BASE_URL;
    const normalizedBaseUrl = baseUrl.endsWith("/") ? baseUrl : `${baseUrl}/`;

    return new URL("responses", normalizedBaseUrl).toString();
  }

  getLangChainAgentSpec(params: {
    maxTokens: number;
    purpose: AgentModelPurpose;
  }) {
    const extraRequestBodyParameters =
      (this.options.extraRequestBodyParameters || {}) as Record<string, unknown> & {
        reasoning?: Record<string, unknown>;
        text?: Record<string, unknown>;
      };
    const { reasoning, ...modelKwargs } = extraRequestBodyParameters;
    const configuredBaseUrl = this.getConfiguredBaseUrl();
    const normalizedModelKwargs = { ...modelKwargs };

    if (configuredBaseUrl) {
      const existingText = normalizedModelKwargs.text as Record<string, unknown> | undefined;

      normalizedModelKwargs.text = existingText?.format
        ? existingText
        : {
            ...existingText,
            format: {
              type: "text",
            },
          };
    }

        const clientConfiguration = this.getClientConfiguration();

    const chatOpenAiOptions: Record<string, unknown> = {
      model: this.options.model || "gpt-5-nano",
      apiKey: this.options.openAiApiKey,
      useResponsesApi: true,
      maxTokens: params.maxTokens,
      reasoning: reasoning ?? {
        effort: getAgentReasoningEffort(params.purpose),
        summary: "detailed",
      },
      modelKwargs: normalizedModelKwargs,
    };


    let supportsResponseContinuation = true;
    if (configuredBaseUrl) {
      chatOpenAiOptions.supportsStrictToolCalling = false;
      supportsResponseContinuation = false;
    } else {
      chatOpenAiOptions.supportsStrictToolCalling = true;
    }

    if (clientConfiguration) {
      chatOpenAiOptions.configuration = clientConfiguration;
    }

    return {
      model: new ChatOpenAI(chatOpenAiOptions as any),
      middleware:
        params.purpose === "primary" && supportsResponseContinuation
          ? [createOpenAiResponsesContinuationMiddleware()]
          : [],
    };
  }

  complete = async (
    requestOrContent: CompletionRequestInput | string,
    maxTokens = 50,
    outputSchema?: any,
    reasoningEffort: ReasoningEffort = "low",
    toolsOrOnChunk?: CompletionTool[] | StreamChunkCallback,
    onChunk?: StreamChunkCallback,
  ): Promise<{
    content?: string;
    finishReason?: string;
    error?: string;
  }> => {
    const request =
      typeof requestOrContent === "string"
        ? {
            content: requestOrContent,
            maxTokens,
            outputSchema,
            reasoningEffort,
            tools: Array.isArray(toolsOrOnChunk) ? toolsOrOnChunk : undefined,
            onChunk:
              typeof toolsOrOnChunk === "function"
                ? toolsOrOnChunk
                : onChunk,
          }
        : requestOrContent;
    const {
      content,
      maxTokens: requestMaxTokens = 50,
      outputSchema: requestOutputSchema,
      reasoningEffort: requestReasoningEffort = "low",
      tools,
      onChunk: streamChunkCallback,
    } = request;
    const model = this.options.model || "gpt-5-nano";
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
      max_output_tokens: requestMaxTokens,
      stream: isStreaming,
      text: requestOutputSchema
        ? {
            format: {
              type: "json_schema",
              ...requestOutputSchema,
            },
          }
        : {
            format: {
              type: "text",
            },
          },
      reasoning: {
        effort: requestReasoningEffort,
        summary: "detailed",
      },
      tools: openAiTools,
      ...extra,
    } as ResponseCreateBody;

    const serializedBody = JSON.stringify(body);

    if (this.shouldDumpRawRequest()) {
      this.dumpRawRequest(this.getResponsesUrl(), serializedBody);
    }

    const resp = await fetch(this.getResponsesUrl(), {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${this.options.openAiApiKey}`,
      },
      body: serializedBody,
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