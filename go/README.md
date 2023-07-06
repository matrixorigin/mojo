# mojo tool, the little tool that could be ...

## Build
`make`

## Run
binary is at build/mojo.  Just run `mojo`

```
Fengs-MacBook-Pro:~ fengttt$ mojo
>>> help

Commands:
  !connect      connect to database
  !delete       delete query
  !insert       insert query
  !mo           genric mo command/sql
  !plot         plot saved query result
  !repro        repro a bug
  !save         save query result
  !select       select query
  !set          set variable
  !show         show variable value
  !test         dev/test only
  !update       update query
  !with         with select query
  .             . starts a multi line chat with gpt, end input with a ;
  .test         chat tester
  clear         clear the screen
  exit          exit the program
  help          display help

>>> !repro help

repro a bug

Commands:
  moc-956      playing with autoincr to repor slowdown

>>> !repro moc-956 -h
Usage of moc-956:
  -b string
    	batch, single, prepare (default "batch")
  -s	single table
  -t string
    	begin, badbegin (default "auto")

>>> !repro moc-956 -s batch
MoC-956: autoincr repro, batchmode:batch, txnmode:auto
drop table table: 0
drop table table: 1
drop table table: 2

```

