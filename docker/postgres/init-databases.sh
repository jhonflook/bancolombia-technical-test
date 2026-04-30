#!/bin/bash
set -e

# Create debitdb database and user using environment variables
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    CREATE DATABASE ${DEBITDB_DB:-debitdb};
    CREATE USER ${DEBITDB_USER:-debit} WITH ENCRYPTED PASSWORD '${DEBITDB_PASSWORD:-debit}';
    GRANT ALL PRIVILEGES ON DATABASE ${DEBITDB_DB:-debitdb} TO ${DEBITDB_USER:-debit};
EOSQL

# Grant schema permissions on debitdb
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "${DEBITDB_DB:-debitdb}" <<-EOSQL
    GRANT ALL ON SCHEMA public TO ${DEBITDB_USER:-debit};
EOSQL

# Create metabase application database
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    CREATE DATABASE metabaseappdb;
    CREATE USER metabase WITH ENCRYPTED PASSWORD 'metabase';
    GRANT ALL PRIVILEGES ON DATABASE metabaseappdb TO metabase;
EOSQL

# Grant schema permissions on metabaseappdb
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "metabaseappdb" <<-EOSQL
    GRANT ALL ON SCHEMA public TO metabase;
EOSQL

# Create MLflow tracking database
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    CREATE DATABASE mlflowdb;
    CREATE USER mlflow WITH ENCRYPTED PASSWORD 'mlflow';
    GRANT ALL PRIVILEGES ON DATABASE mlflowdb TO mlflow;
EOSQL

# Grant schema permissions on mlflowdb
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "mlflowdb" <<-EOSQL
    GRANT ALL ON SCHEMA public TO mlflow;
EOSQL
