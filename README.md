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

OpenAI-compatible providers can be used by overriding the base URL:

```ts
const adapter = new CompletionAdapterOpenAIResponses({
	openAiApiKey: process.env.OVH_AI_ENDPOINTS_ACCESS_TOKEN as string,
	baseUrl: "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1",
	model: "gpt-oss-20b",
	extraRequestBodyParameters: {
		store: false,
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
- Responses API continuation with `previousResponseId`

## Responses continuation

For regular completion flows you can pass the previous Responses API id to reuse
server-side context. The adapter returns the current `responseId`, which can be
used as `previousResponseId` on the next call:

```ts
const first = await adapter.complete({
	content: "Summarize the project requirements",
	maxTokens: 300,
});

const second = await adapter.complete({
	content: "Now turn that into three implementation milestones",
	maxTokens: 300,
	previousResponseId: first.responseId,
});
```
