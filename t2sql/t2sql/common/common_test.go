package common

import (
	"strings"
	"testing"
)

func TestProjectRoot(t *testing.T) {
	root := ProjectRoot()
	if !strings.HasSuffix(root, "mojo/t2sql") {
		t.Errorf("ProjectRoot() = %s, want %s", root, "mojo/t2sql")
	}
}
