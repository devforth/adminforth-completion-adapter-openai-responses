import type OpenAI from "openai";

export type AdapterExtraRequestBodyParameters = Partial<
  Omit<OpenAI.Responses.ResponseCreateParamsNonStreaming, "stream">
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
   * Forces LangChain agent mode to use the Chat Completions API instead of the
   * Responses API.
   *
   * When omitted, the adapter keeps the current default behavior:
   * - official OpenAI uses the Responses API
   * - custom `baseUrl` providers use the Chat Completions API
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
