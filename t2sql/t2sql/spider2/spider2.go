package spider2

import (
	"bufio"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"

	"github.com/matrixorigin/mojo/t2sql/t2sql/common"
)

type ColInfoParsed struct {
	TableName   string   `json:"table_name"`
	Colnames    []string `json:"column_names"`
	Coltypes    []string `json:"column_types"`
	Description []string `json:"description"`
}

func loadOneDbInfo(dbInfoDir string, sqliteDir string, dbName string) (*common.DbInfo, error) {
	dbDir := filepath.Join(dbInfoDir, dbName)
	files, err := os.ReadDir(dbDir)
	if err != nil {
		return nil, fmt.Errorf("failed to read dbDir: %s, %w", dbDir, err)
	}

	dbInfo := common.DbInfo{
		Name:    dbName,
		SqlLite: filepath.Join(sqliteDir, dbName+".sqlite"),
	}

	switch dbName {
	case "DB_IMDB":
		dbInfo.SqlLite = filepath.Join(sqliteDir, "Db-IMDB.sqlite")
	case "SQLITE_SAKILA":
		dbInfo.SqlLite = filepath.Join(sqliteDir, "sqlite-sakila.sqlite")
	}

	for _, file := range files {
		if strings.HasSuffix(file.Name(), ".json") {
			jsonFile := filepath.Join(dbDir, file.Name())
			jsonData, err := os.ReadFile(jsonFile)
			if err != nil {
				return nil, fmt.Errorf("cannot read json file: %s, %w", jsonFile, err)
			}

			// replace NaN with null, NaN is not a valid json value
			jsonData = []byte(strings.ReplaceAll(string(jsonData), ": NaN", ": null"))

			var colParse ColInfoParsed
			if err := json.Unmarshal(jsonData, &colParse); err != nil {
				return nil, fmt.Errorf("cannot unmarshal json file: %s, %w", jsonFile, err)
			}

			var tableInfo common.TableInfo
			tableInfo.OrigName = colParse.TableName
			tableInfo.Name = colParse.TableName
			switch strings.ToLower(tableInfo.Name) {
			case "match":
				tableInfo.Name = "match_table"
			}

			tableInfo.Sql = fmt.Sprintf("CREATE TABLE %s (", tableInfo.Name)
			for i, colname := range colParse.Colnames {
				coltype := strings.ToLower(colParse.Coltypes[i])
				switch colname {
				case "rank", "index", "table", "column", "group", "range",
					"cross", "change":
					colname = fmt.Sprintf("%s_%s", tableInfo.Name, colname)
				}

				if strings.HasPrefix(colname, "Unnamed: ") {
					colname = fmt.Sprintf("unnamed_%s", colname[len("Unnamed: "):])
				}

				colname = strings.ReplaceAll(colname, "%", "percent")
				colname = strings.ReplaceAll(colname, "/", "_over_")
				colname = strings.ReplaceAll(colname, "(", "_")
				colname = strings.ReplaceAll(colname, ")", "_")
				colname = strings.ReplaceAll(colname, "-", "_")

				if strings.HasSuffix(colname, "%") {
					colname = fmt.Sprintf("%s_percent", colname[:len(colname)-1])
				}

				switch coltype {
				case "num":
					// this NUM thing is in the f1 database, it is a datetime, or time.
					// many other _date/time columns in f1 database used text type, so,
					// we use text type too.
					coltype = "text"
				case "", "blob sub_type text", "point":
					coltype = "text"
				case "integer":
					switch colname {
					case "height", "weight":
						// some measurements are not integers.
						coltype = "real"
					default:
						// some data has int overflow, use bigint to be safe.
						coltype = "bigint"
					}
				case "jsonb":
					coltype = "json"
				case "timestamp with time zone":
					coltype = "timestamp"
				}

				if strings.HasPrefix(coltype, "nvarchar") {
					coltype = "text"
				}

				if strings.HasPrefix(coltype, "character") {
					coltype = "varchar(255)"
				}

				// numeric/decicaml should be fine, but we convert it to real due to
				// https://github.com/matrixorigin/matrixone/issues/22396
				if strings.HasPrefix(coltype, "numeric") || strings.HasPrefix(coltype, "decimal") {
					coltype = "real"
				}

				tableInfo.Sql += fmt.Sprintf("%s %s", colname, coltype)
				if i == len(colParse.Colnames)-1 {
					if dbName == "f1" && (tableInfo.Name == "lap_times" || tableInfo.Name == "pit_stops") {
						tableInfo.Sql += ",\n"
						tableInfo.ColInfos = append(tableInfo.ColInfos, common.ColInfo{
							Name:        colname,
							Type:        coltype,
							OrigName:    colParse.Colnames[i],
							OrigType:    colParse.Coltypes[i],
							Description: colParse.Description[i],
						})
						tableInfo.Sql += "seconds float"
						colname = "seconds"
						coltype = "float"
					}
					tableInfo.Sql += ");\n"
				} else {
					tableInfo.Sql += ",\n"
				}

				tableInfo.ColInfos = append(tableInfo.ColInfos, common.ColInfo{
					Name:        colname,
					Type:        coltype,
					OrigName:    colParse.Colnames[i],
					OrigType:    colParse.Coltypes[i],
					Description: colParse.Description[i],
				})

				// add an extra column full_name
				if dbName == "f1" && tableInfo.Name == "drivers" && colname == "surname" {
					tableInfo.Sql += "full_name text,\n"
					tableInfo.ColInfos = append(tableInfo.ColInfos, common.ColInfo{
						Name:        "full_name",
						Type:        "text",
						OrigName:    "full_name",
						OrigType:    "text",
						Description: "full name",
					})
				}
			}
			dbInfo.TableInfos = append(dbInfo.TableInfos, tableInfo)
		}
	}
	return &dbInfo, nil
}

func Spider2LoadDbInfo() (map[string]*common.DbInfo, error) {
	dbInfos := make(map[string]*common.DbInfo)

	rootDir := common.ProjectRoot()
	dbInfoDir := filepath.Join(rootDir, "repo3/Spider2/spider2-lite/resource/databases/sqlite")
	sqliteDir := filepath.Join(rootDir, "repo3/data/spider2")
	files, err := os.ReadDir(dbInfoDir)
	if err != nil {
		return nil, fmt.Errorf("failed to read dbInfoDir: %s, %w", dbInfoDir, err)
	}

	for _, file := range files {
		dbInfo, err := loadOneDbInfo(dbInfoDir, sqliteDir, file.Name())
		if err != nil {
			return nil, err
		}
		dbInfos[dbInfo.Name] = dbInfo
	}

	return dbInfos, nil
}

func Spider2CreateMoTables(dbInfo *common.DbInfo) error {
	dbName := dbInfo.Name
	mo, err := common.OpenMoDB()
	if err != nil {
		return fmt.Errorf("failed to open mo db: %s, %w", dbName, err)
	}
	defer mo.Close()

	common.MustExec(mo, "DROP DATABASE IF EXISTS "+dbName)
	common.MustExec(mo, "CREATE DATABASE "+dbName)
	common.MustExec(mo, "USE "+dbName)

	for _, tableInfo := range dbInfo.TableInfos {
		_, err = mo.Exec(tableInfo.Sql)
		if err != nil {
			return fmt.Errorf("failed to create db %s, table %s: %w", dbName, tableInfo.Name, err)
		}
	}
	return nil
}

func LoadMoDB(dbInfo *common.DbInfo) error {
	mo, err := common.OpenMoDB()
	if err != nil {
		return fmt.Errorf("failed to open mo db: %s, %w", dbInfo.Name, err)
	}
	defer mo.Close()

	common.MustExec(mo, "USE "+dbInfo.Name)

	for _, tableInfo := range dbInfo.TableInfos {
		log.Printf("Loading db %s table %s -> %s\n", dbInfo.Name, tableInfo.OrigName, tableInfo.Name)
		rows, err := common.ReadSqliteRows(dbInfo.SqlLite, "SELECT * FROM \""+tableInfo.OrigName+"\"")
		if err != nil {
			return err
		}

		stmt := "INSERT INTO " + tableInfo.Name + " VALUES ("
		for i := range tableInfo.ColInfos {
			stmt += "?"
			if i < len(tableInfo.ColInfos)-1 {
				stmt += ","
			}
		}
		stmt += ");"

		rowBuf := make([]any, len(tableInfo.ColInfos))

		tx, err := mo.Begin()
		prepareStmt, err := tx.Prepare(stmt)
		if err != nil {
			tx.Rollback()
			return fmt.Errorf("cannot prepare statement: %s, %w", stmt, err)
		}
		defer prepareStmt.Close()

		for _, row := range rows {
			for colIdx, col := range row {
				tmpStr, ok := common.ConvertSqliteToMo(col, tableInfo.ColInfos[colIdx].Type)
				if !ok {
					rowBuf[colIdx] = nil
				} else if tmpStr == "" {
					rowBuf[colIdx] = nil
				} else {
					rowBuf[colIdx] = tmpStr
				}
			}
			_, err = prepareStmt.Exec(rowBuf...)
			if err != nil {
				tx.Rollback()
				return fmt.Errorf("cannot exec statement: %s, %v,%w", stmt, row, err)
			}
		}
		tx.Commit()
		log.Printf("Done load db %s table %s, %d rows\n", dbInfo.Name, tableInfo.OrigName, len(rows))

	}
	return nil
}

type Spider2QueryParsed struct {
	Id                string `json:"instance_id"`
	Db                string `json:"db"`
	Question          string `json:"question"`
	ExternalKnowledge string `json:"external_knowledge"`
}

func LoadQueries() error {
	rootDir := common.ProjectRoot()
	jsonFilePath := filepath.Join(rootDir, "repo3/Spider2/spider2-lite/spider2-lite.jsonl")
	extKnowledgeDir := filepath.Join(rootDir, "repo3/Spider2/spider2-lite/resource/")
	goldDir := filepath.Join(rootDir, "repo3/Spider2/spider2-lite/evaluation_suite/gold/sql")

	mo, err := common.OpenMoDB()
	if err != nil {
		return fmt.Errorf("failed to open mo db: %w", err)
	}
	defer mo.Close()

	common.MustExec(mo, "DROP DATABASE IF EXISTS spider2")
	common.MustExec(mo, "CREATE DATABASE spider2")
	common.MustExec(mo, "USE spider2")

	common.MustExec(mo, "DROP STAGE IF EXISTS ext_knowledge")
	common.MustExec(mo, "CREATE STAGE ext_knowledge URL = 'file://"+extKnowledgeDir+"'")
	common.MustExec(mo, `CREATE TABLE queries (id varchar(100) not null primary key , 
	                            db varchar(100), question text, 
	                            external_knowledge datalink, 
								sqlite_gold text,
								sf_gold text
		            )`)

	jsonFile, err := os.Open(jsonFilePath)
	if err != nil {
		return fmt.Errorf("failed to open jsonl file: %s, %w", jsonFilePath, err)
	}
	defer jsonFile.Close()

	stmt := `INSERT INTO queries (id, db, question, external_knowledge, sqlite_gold, sf_gold) 
	                     VALUES (?, ?, ?, cast(cast(? as varchar) as datalink), ?, ?)`
	prepareStmt, err := mo.Prepare(stmt)
	if err != nil {
		return fmt.Errorf("failed to prepare statement: %s, %w", stmt, err)
	}
	defer prepareStmt.Close()

	count := 0
	countGold := 0
	countSfGold := 0

	scanner := bufio.NewScanner(jsonFile)
	for scanner.Scan() {
		line := scanner.Bytes()
		var query Spider2QueryParsed
		err := json.Unmarshal(line, &query)
		if err != nil {
			return fmt.Errorf("failed to unmarshal jsonl file: %s, %w", jsonFilePath, err)
		}

		var goldSql, sfGoldSql string
		sqliteGoldFile := filepath.Join(goldDir, query.Id+".sql")
		if _, err := os.Stat(sqliteGoldFile); err == nil {
			sqliteGold, err := os.ReadFile(sqliteGoldFile)
			if err == nil {
				goldSql = string(sqliteGold)
				countGold++
			} else {
				log.Printf("failed to read sqlite gold file: %s, %v", sqliteGoldFile, err)
			}
		}
		sfGoldFile := filepath.Join(goldDir, "sf_"+query.Id+".sql")
		if _, err := os.Stat(sfGoldFile); err == nil {
			sfGold, err := os.ReadFile(sfGoldFile)
			if err == nil {
				sfGoldSql = string(sfGold)
				countSfGold++
			} else {
				log.Printf("failed to read sf gold file: %s, %v", sfGoldFile, err)
			}
		}

		var pExt any
		if query.ExternalKnowledge != "" {
			pExt = "stage://ext_knowledge/documents/" + query.ExternalKnowledge
		}
		_, err = prepareStmt.Exec(query.Id, query.Db, query.Question, pExt, goldSql, sfGoldSql)
		if err != nil {
			return fmt.Errorf("failed to exec statement: %s, %w", stmt, err)
		}
		count++
	}

	log.Printf("loaded %d queries, %d gold, %d sf gold", count, countGold, countSfGold)
	return nil
}

func LoadMoGold() error {
	mo, err := common.OpenMoDB()
	if err != nil {
		return fmt.Errorf("failed to open mo db: %w", err)
	}
	defer mo.Close()

	dbInfos, err := Spider2LoadDbInfo()
	if err != nil {
		return fmt.Errorf("failed to load db info: %w", err)
	}

	common.MustExec(mo, "USE spider2")
	common.MustExec(mo, "DROP TABLE IF EXISTS mo_gold")
	common.MustExec(mo, "CREATE TABLE mo_gold (id varchar(100) not null primary key, db varchar(100), gold text, err text)")

	rows, err := mo.Query("SELECT id, db, sqlite_gold FROM queries")
	if err != nil {
		return fmt.Errorf("failed to query: %w", err)
	}
	defer rows.Close()

	moIns, err := common.OpenMoDBName("spider2")
	if err != nil {
		return fmt.Errorf("failed to open mo db: %w", err)
	}
	defer moIns.Close()

	for rows.Next() {
		var id, db, sqliteGold string
		err := rows.Scan(&id, &db, &sqliteGold)
		if err != nil {
			return fmt.Errorf("failed to scan: %w", err)
		}

		if sqliteGold == "" {
			common.MustExec(moIns, "INSERT INTO mo_gold (id, db, gold, err) VALUES (?, ?, '', null)", id, db)
			continue
		}

		if dbInfos[db] == nil {
			common.MustExec(moIns, "INSERT INTO mo_gold (id, db, gold, err) VALUES (?, ?, '', 'db not found')", id, db)
			continue
		}

		// try to run sqliteGold, to see if it is valid
		err = common.CheckSyntax(mo, db, sqliteGold)
		if err == nil {
			// valid, insert into mo_gold
			common.MustExec(moIns, "INSERT INTO mo_gold (id, db, gold, err) VALUES (?, ?, ?, null)", id, db, sqliteGold)
		} else {
			moGold, err := common.FixSyntax(mo, db, sqliteGold, err)
			if err != nil {
				common.MustExec(moIns, "INSERT INTO mo_gold (id, db, gold, err) VALUES (?, ?, ?, ?)", id, db, sqliteGold, err.Error())
			} else {
				common.MustExec(moIns, "INSERT INTO mo_gold (id, db, gold, err) VALUES (?, ?, ?, null)", id, db, moGold)
			}
		}
	}
	return nil
}
