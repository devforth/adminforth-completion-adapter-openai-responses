# @adminforth/completion-adapter-openai-responses

AdminForth completion adapter for the OpenAI Responses API.

This package is the fully compatible successor to `@adminforth/completion-adapter-open-ai-chat-gpt`.

## Installation

```bash
pnpm i @adminforth/completion-adapter-openai-responses
```

## Usage

```ts
import CompletionAdapterOpenAIResponses from "@adminforth/completion-adapter-openai-responses";

const adapter = new CompletionAdapterOpenAIResponses({
	openAiApiKey: process.env.OPENAI_API_KEY as string,
	model: "gpt-5-nano",
	extraRequestBodyParameters: {
		temperature: 0.7,
	},
});
```

The adapter supports:

- regular text completion
- `json_schema` structured output
- reasoning effort control
- tool calls
- streaming output chunks
- streaming reasoning chunks
