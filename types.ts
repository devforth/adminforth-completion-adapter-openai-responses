import type OpenAI from "openai";

export type AdapterExtraRequestBodyParameters = Partial<
  Omit<OpenAI.Responses.ResponseCreateParamsNonStreaming, "stream">
> &
  Partial<
    Omit<OpenAI.Chat.Completions.ChatCompletionCreateParamsNonStreaming, "stream">
  >;

export interface AdapterOptions {
  /**
   * OpenAI API key. Go to https://platform.openai.com/, go to Dashboard -> API keys -> Create new secret key
   * Paste value in your .env file OPENAI_API_KEY=your_key
   * Set openAiApiKey: process.env.OPENAI_API_KEY to access it
   */
  openAiApiKey: string;

  /**
   * Optional OpenAI-compatible base URL.
   *
   * Example: `https://oai.endpoints.kepler.ai.cloud.ovh.net/v1`
   */
  baseUrl?: string;

  /**
   * Forces the adapter to use the Chat Completions API instead of the Responses
   * API.
   *
   * When omitted, the adapter uses the Responses API, including for custom
   * `baseUrl` providers.
   */
  useComplitionApi?: boolean;

  /**
   * Model name. Go to https://platform.openai.com/docs/models, select model and copy name.
   * Default is `gpt-5-nano`.
   */
  model?: string;

  /**
   * Additional request body parameters to include in the API request.
   */
  extraRequestBodyParameters?: AdapterExtraRequestBodyParameters;

  /**
   * Logs the exact JSON body sent to the OpenAI Responses endpoint.
   * Authorization headers are not logged.
   */
  dumpRawRequest?: boolean;
}
