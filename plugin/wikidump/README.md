# Wikidump in Web Assembly

## Two functions are implemented for wikidump files.

1. To extract the offset and size of each bzip2 data chunks.  

```
Function Name: 'get_index'
Input: The content of the bzip2 index file in []byte
Return:  JSON array with Object {"offset": offset, "size": size}
```

2. To get all pages in bzip2 data chunk

```
Function Name: 'get_pages'
Input: Single bzip2 data chunk of the multi-stream wikidump file  in []byte
Return: JSON array with pages found in wikidump except redirected pages
```

## To integrate with MatrixOne Database,

Just run the following SQL,

```

mysql> create stage mystage URL='file:///tmp/';

mysql> select * from moplugin_table('https://github.com/matrixone/mojo/raw/main/plugin/wikidump/wikidump.wasm', 'get_index', null, cast('stage://mystage/wiki/index.txt.bz2' as datalink)) as f limit 5;
+--------------------------------------+
| result                               |
+--------------------------------------+
| {"offset": 557, "size": 707018}      |
| {"offset": 707575, "size": 1555786}  |
| {"offset": 2263361, "size": 1508609} |
| {"offset": 3771970, "size": 1011715} |
| {"offset": 4783685, "size": 1232327} |
+--------------------------------------+
5 rows in set (1.18 sec)


mysql> select json_extract(result, "$.revision.text") from moplugin_table('https://github.com/matrixone/mojo/raw/main/plugin/wikidump/wikidump.wasm', 
'get_pages', null, cast('stage://mystage/wiki/wiki.bz2?offset=557&size=707018' as datalink)) as f;

...


mysql> drop stage mystage;

```

