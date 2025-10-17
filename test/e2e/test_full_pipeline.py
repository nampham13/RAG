from llm.LLM_API import call_gemini
from llm.chat_handler import build_messages
from pipeline.pipeline_qa import fetch_retrieval

# Test full pipeline như trong web app
query = 'Query Rewriting'
print(f'Testing full pipeline with query: "{query}"')

# Get context - giống như web app
ret = fetch_retrieval(query, top_k=10, max_chars=8000)  # Tăng lên 8000
context = ret.get('context', '')
sources = ret.get('sources', [])

print(f'Context length: {len(context)}')
print(f'Number of sources: {len(sources)}')

# Check if Query Rewriting is in context
has_query_rewriting_in_context = 'Query Rewriting' in context
print(f'Query Rewriting in context: {has_query_rewriting_in_context}')

if has_query_rewriting_in_context:
    # Show where it appears
    start = context.find('Query Rewriting')
    context_snippet = context[max(0, start-50):start+150]
    print(f'Context snippet: ...{context_snippet}...')
else:
    print('Query Rewriting NOT found in context!')
    print('Context preview:', context[:300] + '...' if len(context) > 300 else context)

# Build messages như web app
messages = build_messages(query=query, context=context, history=[])
print(f'Messages count: {len(messages)}')

# Check the user message content
if messages:
    user_content = messages[-1]['content']  # Last message should be user
    print(f'User message length: {len(user_content)}')
    has_query_rewriting_in_prompt = 'Query Rewriting' in user_content
    print(f'Query Rewriting in final prompt: {has_query_rewriting_in_prompt}')

print('\n=== Testing Gemini API call ===')
try:
    response = call_gemini(messages)
    print(f'Response length: {len(response)}')
    print(f'Response preview: {response[:300]}...' if len(response) > 300 else response)

    # Check if response mentions Query Rewriting
    mentions_query_rewriting = 'Query Rewriting' in response or 'query rewriting' in response.lower()
    print(f'Response mentions Query Rewriting: {mentions_query_rewriting}')

except Exception as e:
    print(f'Error calling Gemini: {e}')