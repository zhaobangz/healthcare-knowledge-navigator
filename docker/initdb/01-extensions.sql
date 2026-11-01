-- Runs once, on first container start. The test database mirrors the main one so
-- integration tests never touch ingested data.
CREATE EXTENSION IF NOT EXISTS vector;
CREATE DATABASE hkn_test OWNER hkn;
\connect hkn_test
CREATE EXTENSION IF NOT EXISTS vector;
