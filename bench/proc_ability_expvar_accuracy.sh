duckdb <<SQL
PRAGMA enable_progress_bar;
INSTALL 'json';
LOAD 'json';

CREATE TABLE results AS SELECT * FROM read_ndjson_auto('$1');

COPY (
    SELECT
    	results.type,
		results.quantity,
		results.pts,
		results.quadgk_err,
		results.lh_len,
		results.val - quadgk.val AS err
    FROM results 
    JOIN
		results as quadgk
	USING (quantity, lh_len)
    WHERE results.type != 'quadgk'
	AND quadgk.type = 'quadgk'
) TO '$2' (FORMAT 'parquet');
SQL
