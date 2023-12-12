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
			Content: "I have a table of nation and its GDP from year 2000 to 2020.    I want to plot the data using gnuplot.   What is the chart type I should use?",
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
