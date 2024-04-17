# Deploying Janus

<!--toc:start-->
- [Deploying Janus](#deploying-janus)
  - [Configuration](#configuration)
    - [Common Configuration](#common-configuration)
      - [Database Connection](#database-connection)
        - [TLS](#tls)
      - [Health Check](#health-check)
      - [Observability](#observability)
        - [Logging](#logging)
        - [Metrics](#metrics)
        - [Tracing](#tracing)
        - [`tokio-console`](#tokio-console)
    - [`aggregator` configuration](#aggregator-configuration)
    - [`garbage_collector` configuration](#garbagecollector-configuration)
    - [`aggregation_job_creator` configuration](#aggregationjobcreator-configuration)
    - [`aggregation_job_driver` configuration](#aggregationjobdriver-configuration)
    - [`collection_job_driver` configuration](#collectionjobdriver-configuration)
  - [Database](#database)
    - [Datastore Keys](#datastore-keys)
    - [Recommended Configuration](#recommended-configuration)
  - [`janus_cli provision-tasks`](#januscli-provision-tasks)
<!--toc:end-->

A full deployment of Janus is composed of multiple Janus components and a
PostgreSQL database. The `aggregator` component is responsible for servicing DAP
requests from other protocol participants, (client, collector, or leader) while
the rest of the components are responsible for advancing the aggregation flow
and collect flow, including making requests to the helper. All communication
between components happens implicitly through the database.

A related binary, `janus_cli`, can be used to perform various one-off tasks, and
interacts with the database as well.

## Configuration

Each Janus component, and `janus_cli`, requires its own set of configuration
parameters. Most configuration is provided via a YAML file, while certain secret
configuration parameters are instead provided on the command line or via
environment variables. The path to the YAML configuration file is provided via
the `--config-file` command line flag. Run a component's binary with `--help`
for a complete list of command line arguments.

### Common Configuration

Certain sections of the configuration file are common to all binaries.

#### Database Connection

Each component must be provided a connection string/URL. Note that the database
password may be separately passed on the command line or through an environment
variable, to override (or fill in) the password set in the configuration file.

##### TLS

By default, TLS is never used for database connections. Opportunistic TLS
connections can be enabled by providing a file with PEM-format CA certificates
to trust. Furthermore, TLS connections can be required by setting
`sslmode=require` in the database connection string.

#### Health Check

Each binary starts an HTTP server to service health check requests from
orchestration systems. The configuration parameter `health_check_listen_address`
determines what socket address this server listens on. Orchestration systems
should send a GET or HEAD request to the path `/healthz`. After a successful
startup, the HTTP server will respond with `200 OK`.

#### Observability

##### Logging

Each Janus component logs to standard output by default. Verbosity can be
controlled by setting the `RUST_LOG` environment variable to a
[filter][EnvFilter]. Depending on whether standard output is a TTY, a
human-readable or structured JSON log format will be chosen automatically. This
can be overridden with the `force_json_output` and `stackdriver_json_output`
configuration parameters under `logging_config`.

The `RUST_LOG` environment variable can be overridden at runtime using the
`/traceconfigz` path on the `health_check_listen_address` endpoint. Here's
an example using this with `curl`:

```bash
$ HEALTH_CHECK_LISTEN_ADDRESS=http://localhost:8000

$ curl $HEALTH_CHECK_LISTEN_ADDRESS/traceconfigz
info

$ curl $HEALTH_CHECK_LISTEN_ADDRESS/traceconfigz -X PUT -d 'debug'
debug
```

The `filter` field corresponds exactly to the [EnvFilter] format.

[EnvFilter]: https://docs.rs/tracing-subscriber/latest/tracing_subscriber/struct.EnvFilter.html

##### Metrics

See the documentation on [configuring metrics](CONFIGURING_METRICS.md) for
detailed instructions.

##### Tracing

See the documentation on [configuring tracing](CONFIGURING_TRACING.md) for
detailed instructions.

##### `tokio-console`

The `tokio-console` tool can be used to monitor the Tokio async runtime. For
detailed instructions, see the documentation on [configuring `tokio-console`
support](CONFIGURING_TOKIO_CONSOLE.md).

### `aggregator` configuration

The `aggregator` component requires a socket address to listen on for DAP
requests, and additional parameters to customize batching of uploaded reports
into database transactions. See the [sample configuration
file](samples/basic_config/aggregator.yaml) for details.

The `aggregator` component can optionally do garbage collection of old data as
well, i.e. when running a single-service helper-only aggregator.

### `garbage_collector` configuration

The `garbage_collector` component requires configuration parameters to determine
how frequently it scans for old data eligible for deletion, how many objects it
deletes at a time, and how much concurrency to use. See the [sample
configuration file](samples/basic_config/garbage_collector.yaml) for details.

### `aggregation_job_creator` configuration

The `aggregation_job_creator` component requires configuration parameters to
determine how frequently it performs its work, and how many reports it will
include in each aggregation job. See the [sample configuration
file](samples/basic_config/aggregation_job_creator.yaml) for details.

### `aggregation_job_driver` configuration

The `aggregation_job_driver` component requires configuration parameters to
determine its schedule for discovering incomplete jobs, maximum per-process
parallelism, duration of leases on jobs, and retry attempts. See the [sample
configuration file](samples/basic_config/aggregation_job_driver.yaml) for
details.

### `collection_job_driver` configuration

The `collection_job_driver` component requires the same set of configuration
parameters as the aggregation job driver above. See the [sample configuration
file](samples/basic_config/collection_job_driver.yaml) for details.

## Database

Janus currently requires PostgreSQL 15. The schema is defined by SQL migration
scripts in the [`db`](../db) directory, which are applied using
[`sqlx`][sqlx-cli]. Initial database setup can be done with `sqlx migrate run`,
using the `--source` argument to point to `janus/db` and providing database
connection information in any of the ways supported by `sqlx` (see its
documentation).

Note that migrations _must_ be applied using `sqlx` as Janus will fail to start
if it cannot locate a `_sqlx_migrations` table to determine whether it supports
the current schema version.

For simple or experimental deployments where the complexity of `sqlx` is not
warranted, it is possible to create a single schema file by concatenating the
`.up.sql` scripts, in order, and applying this schema to the database. When
using this technique, `check_schema_version: false` must be set in each
configuration file. Note that such deployments will not easily be able to
migrate to later versions of the schema, so this technique is likely not
appropriate for deployments which need to retain data across deployments.

Janus also provides a container image called `janus_db_migrator` that makes it
easier to apply SQL migrations in many deployment environments.
`janus_db_migrator` contains the `sqlx` binary as well as the migration scripts
from the corresponding Janus version in the `/migrations` directory inside the
container so that deployments do not have to fetch the migrations from somewhere
else. The image's entrypoint simply invokes `sqlx` so that deployments can pass
the appropriate database configuration and subcommands. See [`sqlx`][sqlx-cli]'s
documentation for more on working with that tool.

Pre-built `janus_db_migrator` images are available at
[us-west2-docker.pkg.dev/divviup-artifacts-public/janus/janus_db_migrator][migrator-images].

[sqlx-cli]: https://crates.io/crates/sqlx-cli
[migrator-images]: https://us-west2-docker.pkg.dev/divviup-artifacts-public/janus

### Datastore Keys

Certain fields in the database are stored in an encrypted form, using
AES-128-GCM. Encryption keys must be provided to each Janus component at runtime
to perform this encryption and decryption. Generate a random 16-byte key using a
cryptographically secure PRNG, encode it using [base64url][base64url], and pass
it via the `DATASTORE_KEYS` environment variable or `--datastore-keys` command
line argument to each process.

Multiple keys can be provided, in order to rotate encryption keys. After
base64url-encoding each key, concatenate them, separated by commas, and pass the
comma separated list through the environment variable or command line argument
as before. The first key in the list is treated as the "primary" key, and will
be used for encrypting all newly-written data. All other keys will only be used
to decrypt data. Note that eager re-encryption of data is not supported yet, so
plan to keep any previous keys in the datastore keys list until all data
encrypted under them has been deleted.

The 16-byte key can be generated with a command like this:

```
openssl rand 16 | basenc --base64url | sed 's/=//g'
```

(If you are missing `basenc` it is part of GNU Coreutils on Linux/macOS.)

[base64url]: https://datatracker.ietf.org/doc/html/rfc4648#section-5

### Recommended Configuration

It is recommended to run Janus on a PostgreSQL instance backed by solid-state
disks.

When using a SSD-backed database, [set `random_page_cost` to `1.1`][random_page_cost].

This can be set in `postgresql.conf`. See the [PostgreSQL documentation][pgdoc]
or your PostgreSQL vendor's documentation for details on how to set this.

[random_page_cost]: https://www.postgresql.org/docs/current/runtime-config-query.html#GUC-RANDOM-PAGE-COST
[pgdoc]: https://www.postgresql.org/docs/current/config-setting.html#CONFIG-SETTING-CONFIGURATION-FILE

## `janus_cli provision-tasks`

Currently, the simplest way to set up DAP tasks inside Janus is via the
`janus_cli provision-tasks` subcommand. As above, `janus_cli` must be provided a
configuration file with database connection information, and it must be provided
the datastore encryption key on the command line or via an environment variable.
In addition, it takes the path to another YAML file as a command line argument,
containing descriptions of tasks' parameters. See
[docs/samples/tasks.yaml](samples/tasks.yaml) for an example of this file's
structure.

If `--generate-missing-parameters` is passed, one or more fields may be omitted
from a task's parameters, and `janus_cli` will fill in randomly generated
values. This is applicable to the task ID, the VDAF verify key, authentication
tokens, and the aggregator HPKE keypair. Depending on which fields are
automatically generated, you may wish to pass `--echo-tasks` as well, to show
what values were used.
