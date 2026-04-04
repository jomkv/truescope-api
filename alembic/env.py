from logging.config import fileConfig

from sqlalchemy import engine_from_config, Connection
from sqlalchemy import pool
try:
    from pgvector import sqlalchemy as pgvector_sa
except ImportError:
    pgvector_sa = None


from alembic import context

from core.db import Base
from core.config import DATABASE_URI
from schemas.article_schema import Article
from schemas.article_chunk_schema import ArticleChunk
from schemas.claim_schema import Claim

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Replace sqlachemy.url with the actual env variable
if DATABASE_URI:
    config.set_main_option("sqlalchemy.url", DATABASE_URI)

# add your model's MetaData object here
# for 'autogenerate' support
# from myapp import mymodel
# target_metadata = mymodel.Base.metadata
target_metadata = Base.metadata

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def do_run_migrations(connection: Connection) -> None:
    # Need to hack the "vector" type into postgres dialect schema types.
    # Otherwise, `alembic check` does not recognize the type
    if connection.dialect.name == "postgresql" and pgvector_sa:
        connection.dialect.ischema_names["vector"] = pgvector_sa.Vector


    context.configure(
        connection=connection,
        target_metadata=target_metadata,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        do_run_migrations(connection)

        # context.configure(connection=connection, target_metadata=target_metadata)

        # with context.begin_transaction():
        #     context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
