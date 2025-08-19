package common

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

type ColInfo struct {
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

			var colInfo ColInfo
			if err := json.Unmarshal(jsonData, &colInfo); err != nil {
				return nil, fmt.Errorf("cannot unmarshal json file: %s, %w", jsonFile, err)
			}

			var tableInfo TableInfo
			tableInfo.Name = colInfo.TableName
			tableInfo.Sql = fmt.Sprintf("CREATE TABLE %s (", tableInfo.Name)

			for i, colname := range colInfo.Colnames {
				coltype := colInfo.Coltypes[i]
				switch colname {
				case "rank":
					colname = fmt.Sprintf("%s_rank", tableInfo.Name)
				}
				if coltype == "" {
					coltype = "text"
				}
				if coltype == "NUM" {
					coltype = "float"
				}

				tableInfo.Sql += fmt.Sprintf("%s %s", colname, coltype)
				if i == len(colInfo.Colnames)-1 {
					tableInfo.Sql += ");\n"
				} else {
					tableInfo.Sql += ",\n"
				}
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
