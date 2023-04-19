duckdb <<SQL
PRAGMA enable_progress_bar;
INSTALL 'json';
LOAD 'json';

CREATE TABLE results AS SELECT * FROM read_ndjson_auto('$1');

COPY (
    SELECT results.slow_lh, results.fast_lh, results.x, results.q, results.r
    FROM results 
    WHERE results.type = 'irf_point'
) TO '$2' (FORMAT 'parquet');

COPY (
    SELECT difficulty,
           discrimination,
           guess,
           slip,
           x_c,
           x_m,
           y_neg_c,
           y_neg_m,
           y_pos_c,
           y_pos_m
    FROM results 
    WHERE results.type = 'item_params'
) TO '$3' (FORMAT 'parquet');
SQL
