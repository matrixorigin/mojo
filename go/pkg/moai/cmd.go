package moai

import (
	"fmt"

	"github.com/abiosoft/ishell/v2"
	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/ebitenutil"
)

type Game struct {
	w, h int
}

func (g *Game) Update() error {
	return nil
}

func (g *Game) Draw(screen *ebiten.Image) {
	ebitenutil.DebugPrint(screen, fmt.Sprintf("TPS: %0.2f, Window Size: %dx%d", ebiten.ActualTPS(), g.w, g.h))
}

func (g *Game) Layout(outsideWidth, outsideHeight int) (int, int) {
	g.w = outsideWidth
	g.h = outsideHeight
	return outsideWidth, outsideHeight
}

func Moai(c *ishell.Context) {
	// ebiten.SetWindowSize(800, 600)
	ebiten.SetWindowTitle("Moai")
	ebiten.SetWindowResizingMode(ebiten.WindowResizingModeEnabled)

	if err := ebiten.RunGame(&Game{}); err != nil {
		c.Println("Error:", err)
	}
	c.Println("Game Over!")
}

// We build a mo command to send generic sql to mo db.
// As shortcut we also build select/insert/update/delete/with
// command to send sql directy.  Note that we only take
// lower case.
func BuildCmd(sh *ishell.Shell) {
	sh.AddCmd(&ishell.Cmd{
		Name: "!x",
		Help: "Start moai UI",
		Func: Moai,
	})
}
