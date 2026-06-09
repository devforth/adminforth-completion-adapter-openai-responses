import type { AdapterOptions } from "./types.js";
import type {
  CompletionAdapter,
  CompletionTool,
} from "adminforth";
import { encoding_for_model, type TiktokenModel } from "tiktoken";
import {
  OpenAIResponsesService,
  type CompletionRequestInput,
  type CompletionResult,
  type ReasoningEffort,
  type StreamChunkCallback,
} from "./openai.js";
import {
  createLangChainAgentSpec,
  type AgentModelPurpose,
} from "./langchain.js";

export type { AdapterOptions } from "./types.js";

class CompletionAdapterOpenAIResponses
  implements CompletionAdapter
{
  options: AdapterOptions;
  private encoding: ReturnType<typeof encoding_for_model>;
  private activeAbortController: AbortController | null = null;
  private openAi: OpenAIResponsesService;

  constructor(options: AdapterOptions) {
    this.options = options;
    this.openAi = new OpenAIResponsesService(options);

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

  abort() {
    this.activeAbortController?.abort();
  }

  isGenerationInProgress() {
    return Boolean(this.activeAbortController);
  }

  private getConfiguredBaseUrl() {
    return this.options.baseUrl;
  }

  private shouldUseComplitionApi() {
    if (typeof this.options.useComplitionApi === "boolean") {
      return this.options.useComplitionApi;
    }

    return false;
  }

  getLangChainAgentSpec(params: {
    maxTokens: number;
    purpose: AgentModelPurpose;
  }) {
    return createLangChainAgentSpec({
      options: this.options,
      maxTokens: params.maxTokens,
      purpose: params.purpose,
      configuredBaseUrl: this.getConfiguredBaseUrl(),
      clientConfiguration: this.openAi.getClientConfiguration(),
      useComplitionApi: this.shouldUseComplitionApi(),
    });
  }

  complete = async (
    requestOrContent: CompletionRequestInput | string,
    maxTokens = 50,
    outputSchema?: any,
    reasoningEffort: ReasoningEffort = "low",
    toolsOrOnChunk?: CompletionTool[] | StreamChunkCallback,
    onChunk?: StreamChunkCallback,
  ): Promise<CompletionResult> => {
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
    const { signal: requestSignal } = request;
    const abortController = new AbortController();
    this.activeAbortController = abortController;
    const abortFromRequestSignal = () => abortController.abort(requestSignal?.reason);

    if (requestSignal?.aborted) {
      abortController.abort(requestSignal.reason);
    } else {
      requestSignal?.addEventListener("abort", abortFromRequestSignal, { once: true });
    }

    try {
      return await this.openAi.complete(request, abortController.signal);
    } finally {
      requestSignal?.removeEventListener("abort", abortFromRequestSignal);
      if (this.activeAbortController === abortController) {
        this.activeAbortController = null;
      }
    }
  };
}

export { CompletionAdapterOpenAIResponses };
export default CompletionAdapterOpenAIResponses;
