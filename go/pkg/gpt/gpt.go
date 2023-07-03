package gpt

import (
	"context"
	"os"
	"strings"

	"github.com/abiosoft/ishell/v2"
	openai "github.com/sashabaranov/go-openai"
)

func Echo(c *ishell.Context) {
	// existing line
	lines := strings.Join(c.RawArgs[1:], " ")
	if len(lines) == 0 || lines[len(lines)-1] != ';' {
		// read multi lines, until a line with only ";" is entered
		moreLines := c.ReadMultiLines(";")
		if len(moreLines) > 0 {
			lines = lines + " " + moreLines[:len(moreLines)-1]
		}
	}

	c.Println("Done reading. You wrote:")
	c.Println(lines)
}

func BuildCmd(sh *ishell.Shell) {
	chatCmd := &ishell.Cmd{
		Name: ".",
		Help: ". starts a multi line chat with gpt, end input with a ;",
		Func: Echo,
	}
	sh.AddCmd(chatCmd)
}

type GptClient struct {
	client *openai.Client
	model  string
}

func NewGptClient(model string) GptClient {
	var client GptClient
	apiKey := os.Getenv("OPENAI_API_KEY")
	client.client = openai.NewClient(apiKey)
	client.model = model
	return client
}

func (cli *GptClient) Chat(msgs []openai.ChatCompletionMessage) (openai.ChatCompletionMessage, error) {
	ctx := context.Background()
	completion, err := cli.client.CreateChatCompletion(ctx, openai.ChatCompletionRequest{
		Model:    cli.model,
		Messages: msgs,
	})

	if err != nil {
		return openai.ChatCompletionMessage{}, err
	}
	return completion.Choices[0].Message, nil
}
