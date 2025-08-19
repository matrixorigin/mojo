package common

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

type ColInfoParsed struct {
	TableName   string   `json:"table_name"`
	Colnames    []string `json:"column_names"`
	Coltypes    []string `json:"column_types"`
	Description []string `json:"description"`
}

func loadOneDbInfo(dbInfoDir string, sqliteDir string, dbName string) (*DbInfo, error) {
	dbDir := filepath.Join(dbInfoDir, dbName)
	files, err := os.ReadDir(dbDir)
	if err != nil {
		return nil, fmt.Errorf("failed to read dbDir: %s, %w", dbDir, err)
	}

	dbInfo := DbInfo{
		Name:    dbName,
		SqlLite: filepath.Join(sqliteDir, dbName+".sqlite"),
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

			var tableInfo TableInfo
			tableInfo.Name = colParse.TableName
			switch strings.ToLower(tableInfo.Name) {
			case "match":
				tableInfo.Name = "match_table"
			}

			tableInfo.Sql = fmt.Sprintf("CREATE TABLE %s (", tableInfo.Name)
			for i, colname := range colParse.Colnames {
				coltype := colParse.Coltypes[i]
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
				case "NUM":
					coltype = "float"
				case "", "BLOB SUB_TYPE TEXT", "point":
					coltype = "text"
				case "jsonb":
					coltype = "json"
				case "timestamp with time zone":
					coltype = "timestamp"
				}

				if strings.HasPrefix(strings.ToLower(coltype), "nvarchar") {
					coltype = "text"
				}

				if strings.HasPrefix(strings.ToLower(coltype), "character") {
					coltype = "varchar(255)"
				}

				tableInfo.Sql += fmt.Sprintf("%s %s", colname, coltype)
				if i == len(colParse.Colnames)-1 {
					tableInfo.Sql += ");\n"
				} else {
					tableInfo.Sql += ",\n"
				}
				tableInfo.ColInfos = append(tableInfo.ColInfos, ColInfo{
					Name:        colname,
					Type:        coltype,
					Description: colParse.Description[i],
				})
			}
			dbInfo.TableInfos = append(dbInfo.TableInfos, tableInfo)
		}
	}
	return &dbInfo, nil
}

func Spider2LoadDbInfo() (map[string]*DbInfo, error) {
	dbInfos := make(map[string]*DbInfo)

	rootDir := ProjectRoot()
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

func Spider2CreateMoTables(dbInfo *DbInfo) error {
	dbName := dbInfo.Name
	mo, err := OpenMoDB()
	if err != nil {
		return fmt.Errorf("failed to open mo db: %s, %w", dbName, err)
	}
	defer mo.Close()

	MustExec(mo, "DROP DATABASE IF EXISTS "+dbName)
	MustExec(mo, "CREATE DATABASE "+dbName)
	MustExec(mo, "USE "+dbName)

	for _, tableInfo := range dbInfo.TableInfos {
		_, err = mo.Exec(tableInfo.Sql)
		if err != nil {
			return fmt.Errorf("failed to create db %s, table %s: %w", dbName, tableInfo.Name, err)
		}
	}
	return nil
}
