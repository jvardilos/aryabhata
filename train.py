import anthropic


def main():
    client = anthropic.Anthropic()

    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=300,
        messages=[{"role": "user", "content": "What is the capital of France?"}],
    )

    print(message.content)


if __name__ == "__main__":
    main()
