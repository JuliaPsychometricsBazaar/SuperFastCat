duckdb $2 <<SQL
PRAGMA enable_progress_bar;
INSTALL 'json';
LOAD 'json';

CREATE TEMP TABLE results AS SELECT * FROM read_ndjson_auto('$1');

CREATE TABLE next_item AS 
    SELECT *
    FROM results 
    WHERE results.type = 'next_item';

CREATE TABLE system AS 
    SELECT *
    FROM results 
    WHERE results.type = 'system';
SQL
