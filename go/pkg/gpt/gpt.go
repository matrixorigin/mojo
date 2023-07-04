package gpt

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"strings"

	"github.com/abiosoft/ishell/v2"
	"github.com/matrixorigin/mojo/pkg/common"
	"github.com/matrixorigin/mojo/pkg/mo"
	openai "github.com/sashabaranov/go-openai"
)

func echoGpt(c *ishell.Context) {
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
		Func: echoGpt,
	}
	sh.AddCmd(chatCmd)

	testCmd := &ishell.Cmd{
		Name: ".test",
		Help: "chat tester",
	}
	testCmd.AddCmd(&ishell.Cmd{
		Name: "echo",
		Help: "Send chat message to gpt and echo output",
		Func: echoGpt,
	})
	testCmd.AddCmd(&ishell.Cmd{
		Name: "plot",
		Help: "Plot a sin curve using altair",
		Func: plotAltair,
	})

	sh.AddCmd(testCmd)
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

func RunAltair(c *ishell.Context, datastr string, chart, chartsrc []string) {
	pysrc := `
import altair as alt
import motr

if __name__ == "__main__":
`
	// next add datastr
	pysrc += datastr + "\n"

	// Then add charts
	for i := 0; i < len(chart); i++ {
		pysrc += chartsrc[i] + "\n"
		pysrc += fmt.Sprintf("\t%s = motr.transform_chart(%s)\n", chart[i], chart[i])
	}

	if len(chart) == 1 {
		pysrc += fmt.Sprintf("\tfinal_chart = %s\n", chart[0])
	} else {
		pysrc += fmt.Sprintf("\tfinal_chart = %s\n", strings.Join(chart, "+"))
	}

	pysrc += fmt.Sprintf("\tfinal_chart.save('%s/mojo_altair.png')\n", common.GetVar("TMP"))

	altFile := common.GetVar("TMP") + "/mojo_altair.py"
	f, err := os.Create(altFile)
	if err != nil {
		c.Println("Error:", err)
	}

	f.WriteString(pysrc)
	f.Close()

	cmd := exec.Command("python3", altFile)
	_, err = cmd.Output()
	if err != nil {
		c.Println("Error:", err)
	}

	displayCmd := exec.Command("imgcat", "-H", "1000px", common.GetVar("TMP")+"/mojo_altair.png")
	output, err := displayCmd.Output()
	if err != nil {
		c.Println("Error:", err)
	}
	c.Println(string(output))
}

func plotAltair(c *ishell.Context) {
	// existing line
	lines := strings.Join(c.RawArgs[2:], " ")
	if len(lines) == 0 || lines[len(lines)-1] != ';' {
		// read multi lines, until a line with only ";" is entered
		moreLines := c.ReadMultiLines(";")
		if len(moreLines) > 0 {
			lines = lines + " " + moreLines[:len(moreLines)-1]
		}
	}

	datastr := fmt.Sprintf(`
	motr.enable("%s")
	data = motr.from_sql("""%s""") 
	`, mo.PyConnStr(), lines)

	chartstr := `
	chart = alt.Chart(data.url()).mark_point().encode(
		x='x:Q',
		y='y:Q'
	)
	`

	RunAltair(c, datastr, []string{"chart"}, []string{chartstr})
}
