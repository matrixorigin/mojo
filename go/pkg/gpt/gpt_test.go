package gpt

import (
	"testing"

	openai "github.com/sashabaranov/go-openai"
)

func TestChat(t *testing.T) {
	client := NewGptClient("gpt-4")

	msgs := []openai.ChatCompletionMessage{
		{
			Role:    "user",
			Content: "Hello!",
		},
	}

	completion, err := client.Chat(msgs)
	if err != nil {
		t.Fatal(err)
	}
	t.Log(completion)
}

func TestAltair(t *testing.T) {

}
