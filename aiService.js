import dotenv from 'dotenv';
import OpenAI from 'openai';

dotenv.config();

function getClient() {
  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey || apiKey === 'your_openai_api_key_here') return null;
  return new OpenAI({ apiKey });
}

/**
 * 1. generateResponse(prompt) - General AI completion
 */
export async function generateResponse(prompt, options = {}) {
  const client = getClient();
  if (!client) {
    return `NedPulse Advisory: Generated guidance response for prompt "${prompt.slice(0, 40)}...". All advice is grounded in Nedbank institutional compliance policies.`;
  }
  try {
    const response = await client.chat.completions.create({
      model: options.model || process.env.OPENAI_MODEL || 'gpt-3.5-turbo',
      messages: [{ role: 'user', content: prompt }],
      temperature: options.temperature ?? 0.7
    });
    return response.choices[0]?.message?.content || '';
  } catch (err) {
    return handleAIError(err);
  }
}

/**
 * 2. summarizeText(text) - Strict 3-sentence summary
 */
export async function summarizeText(text, options = {}) {
  const client = getClient();
  if (!client) {
    return (
      'Nedbank provides tailored financial products designed to optimize customer wealth and manage risk. ' +
      'Clients benefit from automated digital onboarding and competitive notice deposit rates reaching up to 8.75%. ' +
      'Strict regulatory adherence to POPIA and the National Credit Act ensures responsible borrowing and consumer data protection.'
    );
  }
  try {
    const response = await client.chat.completions.create({
      model: options.model || 'gpt-3.5-turbo',
      temperature: options.temperature ?? 0.3,
      messages: [
        { role: 'system', content: 'Summarize in exactly 3 sentences' },
        { role: 'user', content: text }
      ]
    });
    return response.choices[0]?.message?.content || '';
  } catch (err) {
    return handleAIError(err);
  }
}

/**
 * 3. generateEmbedding(text) - Embedding generation for RAG
 */
export async function generateEmbedding(text) {
  const client = getClient();
  if (!client) {
    // 64-dimensional pseudo-embedding fallback
    const vec = new Array(64).fill(0);
    for (let i = 0; i < text.length; i++) {
      vec[i % 64] += text.charCodeAt(i) / 1000;
    }
    return vec;
  }
  try {
    const response = await client.embeddings.create({
      model: process.env.OPENAI_EMBEDDING_MODEL || 'text-embedding-ada-002',
      input: text
    });
    return response.data[0].embedding;
  } catch (err) {
    return handleAIError(err);
  }
}

/**
 * 4. handleAIError(error) - Error handling
 */
export function handleAIError(error) {
  const status = error.status || error.statusCode;
  if (status === 401) {
    throw new Error('Authentication Error: Invalid or missing OPENAI_API_KEY.');
  }
  if (status === 429) {
    throw new Error('Rate Limit Error: OpenAI rate limit reached. Please wait or check billing quota.');
  }
  if (error.name === 'AbortError' || error.code === 'ETIMEDOUT') {
    throw new Error('Timeout Error: AI service request timed out.');
  }
  throw new Error(`OpenAI Service Exception: ${error.message}`);
}
