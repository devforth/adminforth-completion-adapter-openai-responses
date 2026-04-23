export interface AdapterOptions {
  /**
   * OpenAI API key. Go to https://platform.openai.com/, go to Dashboard -> API keys -> Create new secret key
   * Paste value in your .env file OPENAI_API_KEY=your_key
   * Set openAiApiKey: process.env.OPENAI_API_KEY to access it
   */
  openAiApiKey: string;

  /**
   * Model name. Go to https://platform.openai.com/docs/models, select model and copy name.
   * Default is `gpt-5-nano`.
   */
  model?: string;

  /**
   * Additional request body parameters to include in the API request.
   */
  extraRequestBodyParameters?: Record<string, unknown>;
}