import type {
  CompletionStreamEvent,
  CompletionTool,
} from "adminforth";
import OpenAI from "openai";
import type { AdapterOptions } from "./types.js";

export type StreamChunkCallback = (
  chunk: string,
  event?: CompletionStreamEvent,
) => void | Promise<void>;

export type ReasoningEffort =
  | "none"
  | "minimal"
  | "low"
  | "medium"
  | "high"
  | "xhigh";

export type CompletionRequestInput = {
  content: string;
  maxTokens?: number;
  outputSchema?: any;
  reasoningEffort?: ReasoningEffort;
  tools?: CompletionTool[];
  onChunk?: StreamChunkCallback;
  signal?: AbortSignal;
  previousResponseId?: string;
};

type ResponseCreateBody = Omit<
  OpenAI.Responses.ResponseCreateParamsNonStreaming,
  "stream"
>;
type ChatCompletionCreateBody = Omit<
  OpenAI.Chat.Completions.ChatCompletionCreateParamsNonStreaming,
  "stream"
>;
type OpenAIResponsesSuccess = OpenAI.Responses.Response;
type OpenAIChatSuccess = OpenAI.Chat.Completions.ChatCompletion;
type OpenAITool = NonNullable<ResponseCreateBody["tools"]>[number];
type OpenAIChatTool = NonNullable<ChatCompletionCreateBody["tools"]>[number];
type OpenAIFunctionCall = Extract<
  OpenAI.Responses.ResponseOutputItem,
  { type: "function_call" }
>;
type OpenAIChatToolCall = Extract<
  NonNullable<OpenAI.Chat.Completions.ChatCompletionMessage["tool_calls"]>[number],
  { type: "function" }
>;
type ReasoningConfig = ResponseCreateBody["reasoning"];

export type UsedTokens = {
  input_uncached: number;
  input_cached: number;
  output: number;
};

export type CompletionResult =
  | {
      content?: string;
      finishReason?: string;
      responseId?: string;
      used_tokens?: UsedTokens;
      error?: undefined;
    }
  | {
      error: string;
      content?: string;
      finishReason?: string;
      responseId?: string;
      used_tokens?: UsedTokens;
    };

const RAW_REQUEST_LOG_PREFIX = "[CompletionAdapterOpenAIResponses] Raw /responses request";

type FetchInput = Parameters<typeof fetch>[0];
type FetchInit = Parameters<typeof fetch>[1];

function extractOutputText(data: OpenAIResponsesSuccess): string {
  return data.output_text || "";
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

function extractUsedTokens(data: OpenAIResponsesSuccess): UsedTokens | undefined {
  const usage = data.usage;
  if (!usage) {
    return undefined;
  }

  const inputCached = usage.input_tokens_details?.cached_tokens ?? 0;

  return {
    input_uncached: Math.max((usage.input_tokens ?? 0) - inputCached, 0),
    input_cached: inputCached,
    output: usage.output_tokens ?? 0,
  };
}

function extractChatUsedTokens(data: OpenAIChatSuccess): UsedTokens | undefined {
  const usage = data.usage;
  if (!usage) {
    return undefined;
  }

  const inputCached = usage.prompt_tokens_details?.cached_tokens ?? 0;

  return {
    input_uncached: Math.max((usage.prompt_tokens ?? 0) - inputCached, 0),
    input_cached: inputCached,
    output: usage.completion_tokens ?? 0,
  };
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

async function executeChatToolCall(
  toolCall: OpenAIChatToolCall,
  tools?: CompletionTool[],
): Promise<string> {
  const tool = tools?.find((candidate) => candidate.name === toolCall.function.name);
  if (!tool) {
    throw new Error(`Tool "${toolCall.function.name}" not found`);
  }

  const toolResult = await tool.handler(JSON.parse(toolCall.function.arguments));
  if (typeof toolResult === "string") return toolResult;
  if (typeof toolResult === "undefined") return "";
  return JSON.stringify(toolResult);
}

async function resolveToolCallResult(params: {
  response: OpenAIResponsesSuccess;
  tools?: CompletionTool[];
  currentContent?: string;
  onChunk?: StreamChunkCallback;
  usedTokens?: UsedTokens;
}): Promise<CompletionResult | null> {
  const toolCall = extractFunctionCall(params.response);
  if (!toolCall) {
    return null;
  }

  try {
    const toolResult = await executeToolCall(toolCall, params.tools);
    if (typeof params.currentContent === "string" && toolResult) {
      const delta = toolResult.startsWith(params.currentContent)
        ? toolResult.slice(params.currentContent.length)
        : toolResult;
      if (delta) {
        await params.onChunk?.(delta, {
          type: "output",
          delta,
          text: toolResult,
        });
      }
    }

    return {
      content: toolResult,
      finishReason: "tool_call",
      responseId: params.response.id,
      used_tokens: params.usedTokens,
    };
  } catch (error: any) {
    return {
      error: error?.message || "Tool execution failed",
      content: params.currentContent || undefined,
      finishReason: "tool_call",
      responseId: params.response.id,
      used_tokens: params.usedTokens,
    };
  }
}

async function handleCompletedResponse(params: {
  response: OpenAIResponsesSuccess;
  tools?: CompletionTool[];
}): Promise<CompletionResult> {
  const usedTokens = extractUsedTokens(params.response);

  const toolCallResult = await resolveToolCallResult({
    response: params.response,
    tools: params.tools,
    usedTokens,
  });
  if (toolCallResult) {
    return toolCallResult;
  }

  return {
    content: extractOutputText(params.response),
    finishReason: params.response.incomplete_details?.reason
      ? params.response.incomplete_details.reason
      : undefined,
    responseId: params.response.id,
    used_tokens: usedTokens,
  };
}

function buildReasoningConfig(params: {
  reasoning?: ReasoningConfig;
  effort: Exclude<ReasoningEffort, "none"> | ReasoningEffort;
}): ReasoningConfig {
  return {
    summary: "auto",
    effort: params.effort,
    ...params.reasoning,
  };
}

function splitExtraRequestBodyParameters(extra: Partial<ResponseCreateBody>) {
  const { reasoning, ...bodyParameters } = extra;

  return {
    reasoning,
    bodyParameters,
  };
}

function mapTools(tools?: CompletionTool[]): ResponseCreateBody["tools"] {
  if (!tools?.length) {
    return undefined;
  }

  return tools.map((tool): OpenAITool => ({
    type: "function",
    name: tool.name,
    description: tool.description,
    parameters: tool.input_schema,
    strict: false,
  }));
}

function mapChatTools(tools?: CompletionTool[]): ChatCompletionCreateBody["tools"] {
  if (!tools?.length) {
    return undefined;
  }

  return tools.map((tool): OpenAIChatTool => ({
    type: "function",
    function: {
      name: tool.name,
      description: tool.description,
      parameters: tool.input_schema,
      strict: false,
    },
  }));
}

function buildResponseBody(params: {
  options: AdapterOptions;
  request: CompletionRequestInput;
}): ResponseCreateBody {
  const {
    content,
    maxTokens: requestMaxTokens = 50,
    outputSchema: requestOutputSchema,
    reasoningEffort: requestReasoningEffort = "low",
    tools,
  } = params.request;
  const {
    reasoning: extraReasoning,
    bodyParameters: extraBodyParameters,
  } = splitExtraRequestBodyParameters(
    params.options.extraRequestBodyParameters ?? {},
  );

  return {
    ...extraBodyParameters,
    model: params.options.model || "gpt-5-nano",
    input: content,
    max_output_tokens: requestMaxTokens,
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
    reasoning: buildReasoningConfig({
      reasoning: extraReasoning,
      effort: requestReasoningEffort,
    }),
    tools: mapTools(tools),
    ...(params.request.previousResponseId
      ? { previous_response_id: params.request.previousResponseId }
      : {}),
  };
}

function buildChatCompletionBody(params: {
  options: AdapterOptions;
  request: CompletionRequestInput;
}): ChatCompletionCreateBody {
  const {
    content,
    maxTokens: requestMaxTokens = 50,
    outputSchema: requestOutputSchema,
    reasoningEffort: requestReasoningEffort = "low",
    tools,
  } = params.request;
  const {
    reasoning: _extraReasoning,
    text: _extraText,
    input: _extraInput,
    max_output_tokens: _extraMaxOutputTokens,
    previous_response_id: _extraPreviousResponseId,
    ...extraBodyParameters
  } = params.options.extraRequestBodyParameters ?? {};

  return {
    ...extraBodyParameters,
    model: params.options.model || "gpt-5-nano",
    messages: [{ role: "user", content }],
    max_completion_tokens: requestMaxTokens,
    reasoning_effort: requestReasoningEffort,
    response_format: requestOutputSchema
      ? {
          type: "json_schema",
          json_schema: requestOutputSchema,
        }
      : undefined,
    tools: mapChatTools(tools),
  } as ChatCompletionCreateBody;
}

async function resolveChatToolCallResult(params: {
  toolCall?: OpenAIChatToolCall;
  tools?: CompletionTool[];
  currentContent?: string;
  onChunk?: StreamChunkCallback;
  usedTokens?: UsedTokens;
  finishReason?: string;
}): Promise<CompletionResult | null> {
  if (!params.toolCall) {
    return null;
  }

  try {
    const toolResult = await executeChatToolCall(params.toolCall, params.tools);
    if (typeof params.currentContent === "string" && toolResult) {
      const delta = toolResult.startsWith(params.currentContent)
        ? toolResult.slice(params.currentContent.length)
        : toolResult;
      if (delta) {
        await params.onChunk?.(delta, {
          type: "output",
          delta,
          text: toolResult,
        });
      }
    }

    return {
      content: toolResult,
      finishReason: "tool_call",
      used_tokens: params.usedTokens,
    };
  } catch (error: any) {
    return {
      error: error?.message || "Tool execution failed",
      content: params.currentContent || undefined,
      finishReason: params.finishReason || "tool_call",
      used_tokens: params.usedTokens,
    };
  }
}

function getChatContent(data: OpenAIChatSuccess): string {
  const content = data.choices[0]?.message.content;
  if (typeof content === "string") {
    return content;
  }

  return "";
}

export class OpenAIResponsesService {
  private client: OpenAI | null = null;

  constructor(private options: AdapterOptions) {}

  getClientConfiguration() {
    const configuredBaseUrl = this.options.baseUrl;
    const debugFetch = this.options.dumpRawRequest === true
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

  async complete(
    request: CompletionRequestInput,
    signal: AbortSignal,
  ): Promise<CompletionResult> {
    if (this.shouldUseCompletionApi()) {
      return this.completeWithChatCompletions(request, signal);
    }

    const { tools, onChunk: streamChunkCallback } = request;
    const isStreaming = typeof streamChunkCallback === "function";
    const body = buildResponseBody({
      options: this.options,
      request,
    });

    let fullContent = "";
    let fullReasoning = "";
    let finishReason: string | undefined;
    let completedResponse: OpenAIResponsesSuccess | undefined;
    let usedTokens: UsedTokens | undefined;

    const handleStreamEvent = async (
      event: OpenAI.Responses.ResponseStreamEvent,
    ) => {
      switch (event.type) {
        case "response.output_text.delta": {
          const delta = event.delta || "";
          if (!delta) return;
          fullContent += delta;
          await streamChunkCallback?.(delta, {
            type: "output",
            delta,
            text: fullContent,
          });
          return;
        }

        case "response.reasoning_summary_text.delta":
        case "response.reasoning_text.delta": {
          const delta = event.delta || "";
          if (!delta) return;
          fullReasoning += delta;
          await streamChunkCallback?.(delta, {
            type: "reasoning",
            delta,
            text: fullReasoning,
          });
          return;
        }

        case "response.completed":
        case "response.incomplete": {
          const response = event.response;
          finishReason =
            response.incomplete_details?.reason || response.status || finishReason;
          completedResponse = response;
          usedTokens = extractUsedTokens(response);
          return;
        }

        case "response.failed":
          throw new Error(event.response.error?.message || "Response failed");

        case "error":
          throw new Error(event.message || "Response failed");
      }
    };

    try {
      if (!isStreaming) {
        const params: OpenAI.Responses.ResponseCreateParamsNonStreaming = {
          ...body,
          stream: false,
        };
        const data = await this.getClient().responses.create(params, { signal });

        return handleCompletedResponse({ response: data, tools });
      }

      const params: OpenAI.Responses.ResponseCreateParamsStreaming = {
        ...body,
        stream: true,
      };
      const stream = await this.getClient().responses.create(params, { signal });

      for await (const event of stream) {
        await handleStreamEvent(event);
      }

      if (completedResponse) {
        const toolCallResult = await resolveToolCallResult({
          response: completedResponse,
          tools,
          currentContent: fullContent,
          onChunk: streamChunkCallback,
          usedTokens,
        });
        if (toolCallResult) {
          return toolCallResult;
        }
      }

      return {
        content: fullContent || undefined,
        finishReason,
        responseId: completedResponse?.id,
        used_tokens: usedTokens,
      };
    } catch (error: any) {
      if (signal.aborted) {
        return {
          error: error?.message || "Generation aborted",
          content: fullContent || undefined,
          finishReason: "aborted",
          used_tokens: usedTokens,
        };
      }

      if (isStreaming) {
        return {
          error: error?.message || "Streaming failed",
          content: fullContent || undefined,
          finishReason,
          used_tokens: usedTokens,
        };
      }

      return {
        error: error?.message || "OpenAI request failed",
      };
    }
  }

  private async completeWithChatCompletions(
    request: CompletionRequestInput,
    signal: AbortSignal,
  ): Promise<CompletionResult> {
    const { tools, onChunk: streamChunkCallback } = request;
    const isStreaming = typeof streamChunkCallback === "function";
    const body = buildChatCompletionBody({
      options: this.options,
      request,
    });

    let fullContent = "";
    let finishReason: string | undefined;
    let usedTokens: UsedTokens | undefined;
    let toolCall: OpenAIChatToolCall | undefined;

    try {
      if (!isStreaming) {
        const params: OpenAI.Chat.Completions.ChatCompletionCreateParamsNonStreaming = {
          ...body,
          stream: false,
        };
        const data = await this.getClient().chat.completions.create(params, { signal });
        const choice = data.choices[0];
        const usedTokens = extractChatUsedTokens(data);
        const toolCallResult = await resolveChatToolCallResult({
          toolCall: choice?.message.tool_calls?.[0] as OpenAIChatToolCall | undefined,
          tools,
          usedTokens,
          finishReason: choice?.finish_reason,
        });
        if (toolCallResult) {
          return toolCallResult;
        }

        return {
          content: getChatContent(data),
          finishReason: choice?.finish_reason,
          used_tokens: usedTokens,
        };
      }

      const params: OpenAI.Chat.Completions.ChatCompletionCreateParamsStreaming = {
        ...body,
        stream: true,
        stream_options: {
          include_usage: true,
          ...(body.stream_options ?? {}),
        },
      };
      const stream = await this.getClient().chat.completions.create(params, { signal });
      const toolCallDeltas = new Map<number, OpenAIChatToolCall>();

      for await (const chunk of stream) {
        if (chunk.usage) {
          usedTokens = extractChatUsedTokens({
            id: chunk.id,
            object: "chat.completion",
            created: chunk.created,
            model: chunk.model,
            choices: [],
            usage: chunk.usage,
          });
        }

        const choice = chunk.choices[0];
        if (!choice) {
          continue;
        }

        finishReason = choice.finish_reason || finishReason;

        const delta = choice.delta.content || "";
        if (delta) {
          fullContent += delta;
          await streamChunkCallback?.(delta, {
            type: "output",
            delta,
            text: fullContent,
          });
        }

        for (const toolCallDelta of choice.delta.tool_calls ?? []) {
          const index = toolCallDelta.index;
          const existing = toolCallDeltas.get(index) ?? {
            id: toolCallDelta.id || "",
            type: "function",
            function: {
              name: "",
              arguments: "",
            },
          };

          toolCallDeltas.set(index, {
            ...existing,
            id: toolCallDelta.id || existing.id,
            type: "function",
            function: {
              name: existing.function.name + (toolCallDelta.function?.name || ""),
              arguments:
                existing.function.arguments +
                (toolCallDelta.function?.arguments || ""),
            },
          });
        }
      }

      toolCall = toolCallDeltas.values().next().value;
      const toolCallResult = await resolveChatToolCallResult({
        toolCall,
        tools,
        currentContent: fullContent,
        onChunk: streamChunkCallback,
        usedTokens,
        finishReason,
      });
      if (toolCallResult) {
        return toolCallResult;
      }

      return {
        content: fullContent || undefined,
        finishReason,
        used_tokens: usedTokens,
      };
    } catch (error: any) {
      if (signal.aborted) {
        return {
          error: error?.message || "Generation aborted",
          content: fullContent || undefined,
          finishReason: "aborted",
          used_tokens: usedTokens,
        };
      }

      if (isStreaming) {
        return {
          error: error?.message || "Streaming failed",
          content: fullContent || undefined,
          finishReason,
          used_tokens: usedTokens,
        };
      }

      return {
        error: error?.message || "OpenAI chat completion request failed",
      };
    }
  }

  private getClient() {
    if (!this.client) {
      this.client = new OpenAI({
        apiKey: this.options.openAiApiKey,
        ...this.getClientConfiguration(),
      });
    }

    return this.client;
  }

  private shouldUseCompletionApi() {
    if (typeof this.options.useCompletionApi === "boolean") {
      return this.options.useCompletionApi;
    }

    return false;
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
}
