import anthropic


def main290():
    print("nobody blink")

def main():
    client = anthropic.Anthropic()

    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=300,
        messages=[{"role": "user", "content": "What is the capital of France?"}],
    )

    print(message.content)


client = anthropic.Anthropic()


# ─────────────────────────────────────────────
# 1. MULTI-TURN CHAT (stateless, you hold state)
# The API remembers nothing — you send full history each time.
# Great for: chatbots, tutors, interview simulators
# ─────────────────────────────────────────────
def chat_loop(system_prompt: str):
    """Simple REPL that maintains conversation history."""
    messages = []

    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break

        messages.append({"role": "user", "content": user_input})

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            system=system_prompt,
            messages=messages,
        )

        reply = response.content[0].text
        messages.append({"role": "assistant", "content": reply})
        print(f"Claude: {reply}")


# ─────────────────────────────────────────────
# 2. CHAIN-OF-THOUGHT PIPELINE
# Each call builds on the last — Claude refines its own output.
# Great for: writing polish, code review → fix, plan → execute
# ─────────────────────────────────────────────
def refine_pipeline(raw_text: str) -> str:
    """Three-stage pipeline: draft → critique → rewrite."""
    messages = [{"role": "user", "content": f"Here is a draft:\n\n{raw_text}"}]

    # Stage 1: ask for critique
    messages.append(
        {"role": "user", "content": "What are the 3 biggest weaknesses in this draft?"}
    )
    critique = client.messages.create(
        model="claude-haiku-4-5-20251001", max_tokens=512, messages=messages
    )
    messages.append({"role": "assistant", "content": critique.content[0].text})

    # Stage 2: rewrite using the critique (Claude sees its own critique!)
    messages.append(
        {"role": "user", "content": "Now rewrite the draft fixing those weaknesses."}
    )
    final = client.messages.create(
        model="claude-haiku-4-5-20251001", max_tokens=1024, messages=messages
    )

    return final.content[0].text


# ─────────────────────────────────────────────
# 3. TOOL USE (agentic loop)
# Claude calls your functions; you execute them and return results.
# Great for: data fetching, calculations, database queries, APIs
# ─────────────────────────────────────────────
def run_agent_with_tools(user_query: str) -> str:
    """Runs a simple tool-use loop until Claude stops calling tools."""
    tools = [
        {
            "name": "get_stock_price",
            "description": "Get the current price of a stock by ticker symbol.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "e.g. AAPL"}
                },
                "required": ["ticker"],
            },
        }
    ]

    messages = [{"role": "user", "content": user_query}]

    while True:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            tools=tools,
            messages=messages,
        )

        if response.stop_reason == "end_turn":
            # Claude is done — extract its final text
            return next(b.text for b in response.content if b.type == "text")

        # Claude wants to call a tool — execute it and feed results back
        messages.append({"role": "assistant", "content": response.content})
        tool_results = []

        for block in response.content:
            if block.type == "tool_use":
                # YOUR actual logic goes here
                result = f"${150.23}"  # e.g. call a real API
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    }
                )

        messages.append({"role": "user", "content": tool_results})


# ─────────────────────────────────────────────
# 4. STRUCTURED OUTPUT EXTRACTION
# Force Claude to return typed JSON — perfect for parsing unstructured data.
# Great for: resumes, invoices, emails, research papers, forms
# ─────────────────────────────────────────────
def extract_invoice(raw_invoice_text: str) -> dict:
    """Extract structured fields from a messy invoice string."""
    from pydantic import BaseModel

    class Invoice(BaseModel):
        vendor: str
        total_amount: float
        currency: str
        line_items: list[str]
        due_date: str

    response = client.messages.parse(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": f"Extract invoice data:\n\n{raw_invoice_text}"}
        ],
        output_format=Invoice,
    )

    return response.parsed_output.model_dump()


# ─────────────────────────────────────────────
# 5. FORKING / BRANCHING CONVERSATIONS
# Snapshot messages at any point and branch into parallel paths.
# Great for: A/B testing prompts, exploring "what if" scenarios,
#            running multiple personas simultaneously
# ─────────────────────────────────────────────
def compare_personas(question: str) -> dict[str, str]:
    """Ask the same question from two different expert perspectives."""
    base = [{"role": "user", "content": question}]

    # Fork the same history with different system prompts
    personas = {
        "optimist": "You are an enthusiastic startup founder. Focus on opportunity.",
        "skeptic": "You are a risk-averse CFO. Focus on what could go wrong.",
    }

    results = {}
    for name, system in personas.items():
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=512,
            system=system,
            messages=base,  # same messages, different system prompt
        )
        results[name] = response.content[0].text

    return results


# ─────────────────────────────────────────────
# 6. SYNTHETIC DATA GENERATION
# Use message history to build variety — each turn adds diversity.
# Great for: test datasets, fine-tuning data, QA examples
# ─────────────────────────────────────────────
def generate_diverse_examples(topic: str, n: int = 5) -> list[str]:
    """Generate N diverse examples, each aware of previous ones to avoid repetition."""
    messages = [
        {
            "role": "user",
            "content": f"Generate a unique example of: {topic}. Be creative.",
        }
    ]
    examples = []

    for i in range(n):
        response = client.messages.create(
            model="claude-haiku-4-5-20251001", max_tokens=256, messages=messages
        )
        example = response.content[0].text
        examples.append(example)

        # Feed it back so next generation avoids similarity
        messages.append({"role": "assistant", "content": example})
        messages.append(
            {
                "role": "user",
                "content": "Good. Now give me another one that's noticeably different.",
            }
        )

    return examples


# ─────────────────────────────────────────────
# 7. STREAMING FOR REAL-TIME UX
# Print tokens as they arrive — no waiting for full response.
# Great for: any user-facing app, long outputs, live demos
# ─────────────────────────────────────────────
def stream_response(prompt: str):
    """Stream tokens to stdout as they're generated."""
    with client.messages.stream(
        model="claude-haiku-4-5-20251001",
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)

    print()  # newline after done
    final = stream.get_final_message()
    print(f"\n[{final.usage.output_tokens} tokens used]")


if __name__ == "__main__":
    main290()
