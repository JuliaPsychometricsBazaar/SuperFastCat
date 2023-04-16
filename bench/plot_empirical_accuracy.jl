using DuckDB
using DBInterface: connect, execute


function main()
    con = connect(DuckDB.DB, ":memory:")
    execute(con, "INSTALL 'json';");
    execute(con, "LOAD 'json';");
    df = execute(con, "SELECT * FROM read_ndjson_auto('outmini.jsonl')")
    println(df)
end


main()