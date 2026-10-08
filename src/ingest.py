import duckdb

con = duckdb.connect("data/fleet.duckdb")

print("Starting scania_readouts...")

con.execute("DROP TABLE IF EXISTS scania_readouts;")
con.execute("CREATE TABLE scania_readouts AS SELECT * FROM read_csv_auto('data/scania/train_operational_readouts.csv');")

print("scania_readouts done")
print("Starting scania_tte...")

con.execute("DROP TABLE IF EXISTS scania_tte;")
con.execute("CREATE TABLE scania_tte AS SELECT * FROM read_csv_auto('data/scania/train_tte.csv');")

print("scania_tte done")
print("Starting scania_specs...")

con.execute("DROP TABLE IF EXISTS scania_specs;")
con.execute("CREATE TABLE scania_specs AS SELECT * FROM read_csv_auto('data/scania/train_specifications.csv');")

print("scania_specs done")

print("scania Readouts:", con.sql("SELECT COUNT(*) FROM scania_readouts").fetchall()[0][0])
print("scania TTE:", con.sql("SELECT COUNT(*) FROM scania_tte").fetchall()[0][0])
print("scania Specifications:", con.sql("SELECT COUNT(*) FROM scania_specs").fetchall()[0][0])

print("scania Readouts:", con.sql("SELECT COUNT(*) FROM (DESCRIBE scania_readouts)").fetchall()[0][0])
print("scania TTE:", con.sql("SELECT COUNT(*) FROM (DESCRIBE scania_tte)").fetchall()[0][0])
print("scania Specifications:", con.sql("SELECT COUNT(*) FROM (DESCRIBE scania_specs)").fetchall()[0][0])

describe = con.sql("DESCRIBE scania_readouts").df()
print(describe[describe['column_name'] == 'time_step'])

con.close()