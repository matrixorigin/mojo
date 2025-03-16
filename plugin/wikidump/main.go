package main

import (
	"bufio"
	"bytes"
	"compress/bzip2"
	"encoding/json"
	"encoding/xml"
	"io"
	"regexp"
	"strconv"
	"strings"
	"time"

	pdk "github.com/extism/go-pdk"
)

// wikidump index format offset:id:title
type IndexLocation struct {
	Offset int `json:"offset"`
	Size   int `json:"size"`
}

// TODO: convert wikitext to plain text by expanding wikipedia templates.
type Stream struct {
	XMLName xml.Name `xml:"stream"`
	Pages   []*Page  `xml:"page"`
}

type Page struct {
	Title    string    `xml:"title" json:"title"`
	Redirect *Redirect `xml:"redirect,omitempty" json:"redirect,omitempty"`
	Revision Revision  `xml:"revision" json:"revision"`
	NS       int64     `xml:"ns" json:"ns"`
	ID       int64     `xml:"id" json:"id"`
}

type Redirect struct {
	Title string `xml:"title,attr" json:"title,attr"`
}

type Revision struct {
	Timestamp   time.Time   `xml:"timestamp" json:"timestamp"`
	Format      string      `xml:"format" json:"format"`
	Text        string      `xml:"text" json:"text"`
	Comment     string      `xml:"comment" json:"comment"`
	Model       string      `xml:"model" json:"model"`
	SHA1        string      `xml:"sha1" json:"sha1"`
	Contributer Contributer `xml:"contributer" json:"contributer"`
	ID          int64       `xml:"id" json:"id"`
	ParentID    int64       `xml:"parentid" json:"parentid"`
}

type Contributer struct {
	Username string `xml:"username" json:"username"`
	ID       int64  `xml:"id" json:"id"`
}

//export get_index
func get_index() int32 {
	byte_flag := false
	data_length := 0
	start_byte := 0

	input := pdk.Input()

	reader := bzip2.NewReader(bytes.NewReader(input))
	scanner := bufio.NewScanner(reader)
	var offsetsize []IndexLocation
	for scanner.Scan() {
		line := scanner.Text()
		rec := strings.SplitN(line, ":", 3)

		offset, err := strconv.Atoi(rec[0])
		if err != nil {
			pdk.SetError(err)
			return 2
		}
		if !byte_flag {
			start_byte = offset
			byte_flag = true
		} else if byte_flag && offset != start_byte {
			data_length = offset - start_byte
			offsetsize = append(offsetsize, IndexLocation{start_byte, data_length})
			start_byte = offset
		}
	}

	b, err := json.Marshal(offsetsize)
	if err != nil {
		pdk.SetError(err)
		return 3
	}
	pdk.Output(b)
	return 0
}

func parseStream(stream []byte) ([]*Page, error) {
	var s Stream
	buff := bytes.NewBufferString("<stream>\n")
	buff.Write(stream)
	buff.WriteString("</stream>")
	err := xml.Unmarshal(buff.Bytes(), &s)
	if err != nil {
		return nil, err
	}
	return s.Pages, nil
}

//export get_pages
func get_pages() int32 {
	input := pdk.Input()

	reader := bzip2.NewReader(bytes.NewReader(input))
	data, err := io.ReadAll(reader)
	if err != nil {
		pdk.SetError(err)
		return 4
	}

	pages, err := parseStream(data)
	if err != nil {
		pdk.SetError(err)
		return 5
	}

	var result []*Page

	for _, p := range pages {
		// ignore redirect page
		if p.Redirect != nil {
			continue
		}
		// convert to plain text
		//p.Revision.Text = ToPlain(p.Revision.Text)
		result = append(result, p)
		//result = append(result, p.Revision.Text)
	}

	jb, err := json.Marshal(result)
	if err != nil {
		pdk.SetError(err)
		return 6
	}
	pdk.Output(jb)
	return 0
}

// We are going to iterate over the characters in text one by one and check multiple condition as to where
// the characters need to be written
// TODO: remove blockquote
func ToPlain(text string) string {
	recTemplate, recLink := false, false
	curBrCount, sqBrCount := 0, 0
	plainIndex := 0
	var linkBuilder, plainBuilder strings.Builder
	for _, char := range text {
		switch char {
		case '\'':
			continue
		case '{':
			if !recTemplate {
				recTemplate = true
			} else {
				curBrCount++
			}
		case '}':
			if curBrCount == 0 {
				recTemplate = false
			} else {
				curBrCount--
			}
		case '[':
			if recTemplate {
			} else if !recLink {
				recLink = true
				linkBuilder.WriteRune('[')
			} else {
				linkBuilder.WriteRune('[')
				sqBrCount++
			}
		case ']':
			if recTemplate {
			} else if sqBrCount == 0 {
				recLink = false
				linkBuilder.WriteRune(']')
				linkString := linkBuilder.String()
				plainBuilder.WriteString(linkDisplay(linkString))
				plainIndex += len(linkString)
				linkBuilder = strings.Builder{}
			} else {
				linkBuilder.WriteRune(char)
				sqBrCount--
			}
		default:
			if recLink {
				linkBuilder.WriteRune(char)
			}
			if !recLink && !recTemplate {
				plainBuilder.WriteRune(char)
				plainIndex++
			}
		}
	}
	return plainBuilder.String()
}

// TODO: handle nested links
func linkDisplay(link string) string {
	re := regexp.MustCompile(`[\[\]]`)
	return re.ReplaceAllLiteralString(link, "")
}

// TODO: trim empty lines and lines with only * in them.
// TODO: split sections.
// TODO: filtering by sections.package main

func main() {}
