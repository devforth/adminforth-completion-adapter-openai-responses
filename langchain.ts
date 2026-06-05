import { AIMessage } from "@langchain/core/messages";
import { ChatOpenAI } from "@langchain/openai";
import { createMiddleware } from "langchain";
import type { AdapterOptions } from "./types.js";

type ReasoningEffort =
  | "none"
  | "minimal"
  | "low"
  | "medium"
  | "high"
  | "xhigh";

export type AgentModelPurpose = "primary" | "summary";

type OpenAiResponsesMetadata = {
  id?: string;
};

type OpenAiResponsesContext = {
  sessionId: string;
  turnId: string;
  abortSignal?: AbortSignal;
};
type ExtraReasoning = NonNullable<
  AdapterOptions["extraRequestBodyParameters"]
>["reasoning"];

function getAgentReasoningEffort(
  purpose: AgentModelPurpose,
): Exclude<ReasoningEffort, "none"> {
  return purpose === "summary" ? "minimal" : "low";
}

function buildReasoningConfig(params: {
  reasoning?: ExtraReasoning;
  effort: Exclude<ReasoningEffort, "none"> | ReasoningEffort;
}) {
  return {
    summary: "auto",
    effort: params.effort,
    ...(params.reasoning ?? {}),
  };
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

export function createLangChainAgentSpec(params: {
  options: AdapterOptions;
  maxTokens: number;
  purpose: AgentModelPurpose;
  configuredBaseUrl?: string;
  clientConfiguration?: Record<string, unknown>;
  useComplitionApi: boolean;
}) {
  const extraRequestBodyParameters =
    params.options.extraRequestBodyParameters || {};
  const { reasoning, ...modelKwargs } = extraRequestBodyParameters;
  const normalizedModelKwargs = { ...modelKwargs };

  const chatOpenAiOptions: Record<string, unknown> = {
    model: params.options.model || "gpt-5-nano",
    apiKey: params.options.openAiApiKey,
    maxTokens: params.maxTokens,
    reasoning: buildReasoningConfig({
      reasoning,
      effort: getAgentReasoningEffort(params.purpose),
    }),
    modelKwargs: normalizedModelKwargs,
  };

  chatOpenAiOptions.useResponsesApi = !params.useComplitionApi;

  let supportsResponseContinuation = true;
  if (params.configuredBaseUrl || params.useComplitionApi) {
    supportsResponseContinuation = false;
  }

  if (params.clientConfiguration) {
    chatOpenAiOptions.configuration = params.clientConfiguration;
  }

  return {
    model: new ChatOpenAI(chatOpenAiOptions as any),
    middleware:
      params.purpose === "primary" && supportsResponseContinuation
        ? [createOpenAiResponsesContinuationMiddleware()]
        : [],
  };
}
