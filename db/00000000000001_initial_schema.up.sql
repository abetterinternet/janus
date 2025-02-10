-- Load pgcrypto for gen_random_bytes.
CREATE EXTENSION pgcrypto;
-- Load an extension to allow indexing over both BIGINT and TSRANGE in a multicolumn GiST index.
CREATE EXTENSION btree_gist;

-- Identifies which aggregator role is being played for this task.
CREATE TYPE AGGREGATOR_ROLE AS ENUM(
    'LEADER',
    'HELPER'
);

-- Specifies the aggregation mode (how aggregation is performed, e.g. synchronously or
-- asynchronously) of a task.
CREATE TYPE AGGREGATION_MODE AS ENUM(
    'SYNCHRONOUS',  -- the Helper handles aggregation synchronously
    'ASYNCHRONOUS'  -- the Helper handles aggregation asynchronously
);

-- Identifies the different types of authentication tokens.
CREATE TYPE AUTH_TOKEN_TYPE AS ENUM(
    'DAP_AUTH', -- DAP-01 style DAP-Auth-Token header
    'BEARER'    -- RFC 6750 bearer token
);

CREATE TYPE HPKE_KEY_STATE AS ENUM(
    'ACTIVE',    -- the key should be advertised to DAP clients
    'PENDING',   -- the key should not be advertised to DAP clients, but could be used for
                 -- decrypting client reports depending on when aggregators pick up the state change
    'EXPIRED'    -- the key is pending deletion. it should not be advertised, but could be used
                 -- for decrypting client reports depending on the age of those reports
);

CREATE TABLE hpke_keys(
    -- These columns should be treated as immutable.
    config_id SMALLINT PRIMARY KEY,  -- HPKE config ID
    config BYTEA NOT NULL,           -- HPKE config, including public key (encoded HpkeConfig message)
    private_key BYTEA NOT NULL,      -- private key (encrypted)

    -- These columns are mutable.
    state HPKE_KEY_STATE NOT NULL DEFAULT 'PENDING',  -- state of the key
    last_state_change_at TIMESTAMP NOT NULL,          -- when the key state was last changed. Used for key rotation logic.

    -- creation/update records
    created_at TIMESTAMP NOT NULL,  -- when the row was created
    updated_at TIMESTAMP NOT NULL,  -- when the row was last changed
    updated_by TEXT NOT NULL        -- the name of the transaction that last updated the row
);

-- Another DAP aggregator who we've partnered with to use the taskprov extension.
CREATE TABLE taskprov_peer_aggregators(
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY, -- artificial ID, internal only.
    endpoint TEXT NOT NULL,              -- peer aggregator HTTPS endpoint
    peer_role AGGREGATOR_ROLE NOT NULL,  -- the role of this peer aggregator

    -- Parameters applied to every task created with this peer aggregator.
    aggregation_mode AGGREGATION_MODE,       -- the aggregation mode in use for tasks with this aggregator (populated when )
    verify_key_init BYTEA NOT NULL,          -- the preshared key used for VDAF verify key derivation.
    tolerable_clock_skew   BIGINT NOT NULL,  -- the maximum acceptable clock skew to allow between client and aggregator, in seconds
    report_expiry_age      BIGINT,           -- the maximum age of a report before it is considered expired (and acceptable for garbage collection), in seconds. NULL means that GC is disabled.
    collector_hpke_config BYTEA NOT NULL,    -- the HPKE config of the collector (encoded HpkeConfig message)

    -- creation/update records
    created_at TIMESTAMP NOT NULL,  -- when the row was created
    updated_by TEXT NOT NULL,       -- the name of the transaction that last updated the row

    CONSTRAINT taskprov_peer_aggregators_endpoint_and_peer_role_unique UNIQUE(endpoint, peer_role)
);

-- Task aggregator auth tokens that we've shared with the peer aggregator.
CREATE TABLE taskprov_aggregator_auth_tokens(
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,  -- artificial ID, internal-only
    peer_aggregator_id BIGINT NOT NULL,  -- task ID the token is associated with
    ord BIGINT NOT NULL,                 -- a value used to specify the ordering of the authentication tokens
    token BYTEA NOT NULL,                -- bearer token used to authenticate messages to/from the other aggregator (encrypted)
    type AUTH_TOKEN_TYPE NOT NULL DEFAULT 'BEARER',

    -- creation/update records
    created_at TIMESTAMP NOT NULL,  -- when the row was created
    updated_by TEXT NOT NULL,       -- the name of the transaction that last updated the row

    CONSTRAINT task_aggregator_auth_tokens_unique_peer_aggregator_id_and_ord UNIQUE(peer_aggregator_id, ord),
    CONSTRAINT fk_peer_aggregator_id FOREIGN KEY(peer_aggregator_id) REFERENCES taskprov_peer_aggregators(id) ON DELETE CASCADE
);

-- Task collector auth tokens that we've shared with the peer aggregator.
CREATE TABLE taskprov_collector_auth_tokens(
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,  -- artificial ID, internal-only
    peer_aggregator_id BIGINT NOT NULL,  -- task ID the token is associated with
    ord BIGINT NOT NULL,                 -- a value used to specify the ordering of the authentication tokens
    token BYTEA NOT NULL,                -- bearer token used to authenticate messages to/from the other aggregator (encrypted)
    type AUTH_TOKEN_TYPE NOT NULL DEFAULT 'BEARER',

    -- creation/update records
    created_at TIMESTAMP NOT NULL,  -- when the row was created
    updated_by TEXT NOT NULL,       -- the name of the transaction that last updated the row

    CONSTRAINT task_collector_auth_tokens_unique_peer_aggregator_id_and_ord UNIQUE(peer_aggregator_id, ord),
    CONSTRAINT fk_peer_aggregator_id FOREIGN KEY(peer_aggregator_id) REFERENCES taskprov_peer_aggregators(id) ON DELETE CASCADE
);

-- Corresponds to a DAP task, containing static data associated with the task.
CREATE TABLE tasks(
    id                          BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,  -- artificial ID, internal-only
    task_id                     BYTEA UNIQUE NOT NULL,     -- 32-byte TaskID as defined by the DAP specification
    aggregator_role             AGGREGATOR_ROLE NOT NULL,  -- the role of this aggregator for this task
    peer_aggregator_endpoint    TEXT NOT NULL,             -- peer aggregator's API endpoint
    batch_mode                  JSONB NOT NULL,            -- the batch mode in use for this task, along with its parameters
    aggregation_mode            AGGREGATION_MODE,          -- the aggregation mode in use for this task (populated for Helper only)
    vdaf                        JSON NOT NULL,             -- the VDAF instance in use for this task, along with its parameters
    task_start                  TIMESTAMP,                 -- the time before which client reports are not accepted
    task_end                    TIMESTAMP,                 -- the time after which client reports are no longer accepted
    report_expiry_age           BIGINT,                    -- the maximum age of a report before it is considered expired (and acceptable for garbage collection), in seconds. NULL means that GC is disabled.
    min_batch_size              BIGINT NOT NULL,           -- the minimum number of reports in a batch to allow it to be collected
    time_precision              BIGINT NOT NULL,           -- the duration to which clients are expected to round their report timestamps, in seconds
    tolerable_clock_skew        BIGINT NOT NULL,           -- the maximum acceptable clock skew to allow between client and aggregator, in seconds
    collector_hpke_config       BYTEA,                     -- the HPKE config of the collector (encoded HpkeConfig message)
    vdaf_verify_key             BYTEA NOT NULL,            -- the VDAF verification key (encrypted)
    taskprov_task_info          BYTEA,                     -- if applicable, the task_info field of a Taskprov TaskConfig structure

    -- Authentication token used to authenticate messages to/from the other aggregator.
    -- These columns are NULL if the task was provisioned by taskprov.
    aggregator_auth_token_type  AUTH_TOKEN_TYPE,    -- the type of the authentication token
    aggregator_auth_token       BYTEA,              -- encrypted bearer token (only set for leader)
    aggregator_auth_token_hash  BYTEA,              -- hash of the token (only set for helper)
    CONSTRAINT aggregator_auth_token_null CHECK (
        -- If aggregator_auth_token_type is not NULL, then exactly one of aggregator_auth_token or
        -- aggregator_auth_token_hash must be not NULL.
        ((aggregator_auth_token_type IS NOT NULL) AND (aggregator_auth_token IS NULL) != (aggregator_auth_token_hash IS NULL))
        -- If aggregator_auth_token_type is NULL, then both aggregator_auth_token and
        -- aggregator_auth_token_hash must be NULL
        OR ((aggregator_auth_token_type IS NULL) AND (aggregator_auth_token IS NULL) AND (aggregator_auth_token_hash IS NULL))
    ),

    -- Authentication token used to authenticate messages to the leader from the collector. These
    -- columns are NULL if the task was provisioned by taskprov or if the task's role is helper.
    collector_auth_token_type   AUTH_TOKEN_TYPE,    -- the type of the authentication token
    collector_auth_token_hash        BYTEA,         -- hash of the token
    -- The collector_auth_token columns must either both be NULL or both be non-NULL
    CONSTRAINT collector_auth_token_null CHECK ((collector_auth_token_type IS NULL) = (collector_auth_token_hash IS NULL)),

    -- creation/update records
    created_at TIMESTAMP NOT NULL,  -- when the row was created
    updated_at TIMESTAMP NOT NULL,  -- when the row was last changed
    updated_by TEXT NOT NULL        -- the name of the transaction that last updated the row
);
CREATE INDEX task_id_index ON tasks(task_id);

-- Per task report upload counters.
--
-- This table is extremely update-heavy and the updates should qualify for the
-- heap-only tuple optimization. Set the fillfactor so that we leave enough space
-- per heap page for HOT updates.
--
-- Setting the exact fillfactor is a tradeoff between disk space and likelihood
-- of HOT updates. We don't want to set fillfactor too low and waste disk space.
-- 50 represents a 2x table size bump, which seems like an acceptible tradeoff.
--
-- See https://www.postgresql.org/docs/current/storage-hot.html
CREATE TABLE task_upload_counters(
    id                     BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,  -- artificial ID, internal-only
    task_id                BIGINT NOT NULL,
    ord                    BIGINT NOT NULL,           -- Index of this task_upload_counters shard.s

    interval_collected     BIGINT NOT NULL DEFAULT 0, -- reports submitted for an interval that was already collected.
    report_decode_failure  BIGINT NOT NULL DEFAULT 0, -- reports which failed to decode.
    report_decrypt_failure BIGINT NOT NULL DEFAULT 0, -- reports which failed to decrypt.
    report_expired         BIGINT NOT NULL DEFAULT 0, -- reports that were older than the task's report_expiry_age.
    report_outdated_key    BIGINT NOT NULL DEFAULT 0, -- reports that were encrypted with an unknown or outdated HPKE key.
    report_success         BIGINT NOT NULL DEFAULT 0, -- reports that were successfully uploaded.
    report_too_early       BIGINT NOT NULL DEFAULT 0, -- reports whose timestamp is too far in the future.
    task_not_started       BIGINT NOT NULL DEFAULT 0, -- reports sent to the task before it started.
    task_ended             BIGINT NOT NULL DEFAULT 0, -- reports sent to the task after it ended.

    CONSTRAINT fk_task_id FOREIGN KEY(task_id) REFERENCES tasks(id) ON DELETE CASCADE,
    CONSTRAINT task_upload_counters_unique UNIQUE(task_id, ord)
) WITH (fillfactor = 50);

-- Per-task report aggregation counters, used for metrics.
--
-- Fillfactor is lowered to improve the likelihood of heap-only tuple optimizations. See the
-- discussion around this setting for the task_upload_counters table.
CREATE TABLE task_aggregation_counters(
    id       BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,  -- artificial ID, internal only
    task_id  BIGINT NOT NULL,                                  -- task ID the counter is associated with
    ord      BIGINT NOT NULL,                                  -- the ordinal index of the task aggregation counter

    success  BIGINT NOT NULL DEFAULT 0,  -- reports successfully aggregated

    CONSTRAINT task_aggregation_counters_unique_id UNIQUE(task_id, ord),
    CONSTRAINT fk_task_id FOREIGN KEY(task_id) REFERENCES tasks(id) ON DELETE CASCADE
) WITH (fillfactor = 50);

-- Individual reports received from clients.
--
-- Reports might be "scrubbed"; this will null out many fields (noted in the individual fields' 
-- comments), keeping the row itself & some basic metadata for use in report replay protection.
CREATE TABLE client_reports(
    id                              BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY, -- artificial ID, internal-only
    task_id                         BIGINT NOT NULL,                 -- task ID the report is associated with
    report_id                       BYTEA NOT NULL,                  -- 16-byte ReportID as defined by the DAP specification
    client_timestamp                TIMESTAMP NOT NULL,              -- report timestamp, from client

    public_extensions               BYTEA,                           -- encoded sequence of public Extension messages (opaque DAP messages, populated for unscrubbed reports only)
    public_share                    BYTEA,                           -- encoded public share (opaque VDAF message, populated for unscrubbed reports only)
    leader_private_extensions       BYTEA,                           -- encoded sequence of leader's private Extension messages (opaque DAP messages, populated for unscrubbed reports only)
    leader_input_share              BYTEA,                           -- encoded, decrypted leader input share (opaque VDAF message, populated for unscrubbed reports only)
    helper_encrypted_input_share    BYTEA,                           -- encoded HpkeCiphertext message containing the helper's input share (opaque DAP message, populated for unscrubbed reports only)

    aggregation_started             BOOLEAN NOT NULL DEFAULT FALSE,  -- has this client report been associated with an aggregation job?

    -- creation/update records
    created_at TIMESTAMP NOT NULL,  -- when the row was created
    updated_at TIMESTAMP NOT NULL,  -- when the row was last changed
    updated_by TEXT NOT NULL,       -- the name of the transaction that last updated the row

    CONSTRAINT client_reports_unique_task_id_and_report_id UNIQUE(task_id, report_id),
    CONSTRAINT fk_task_id FOREIGN KEY(task_id) REFERENCES tasks(id) ON DELETE CASCADE
);
CREATE INDEX client_reports_task_and_timestamp_unaggregated_index ON client_reports (task_id, client_timestamp) WHERE aggregation_started = FALSE;
CREATE INDEX client_reports_task_and_timestamp_index ON client_reports(task_id, client_timestamp);

-- Specifies the possible state of an aggregation job.
CREATE TYPE AGGREGATION_JOB_STATE AS ENUM(
    'ACTIVE',            -- at least one included report is in a non-terminal state, processing can continue at will
    'AWAITING_REQUEST',  -- at least one included report is in a non-terminal state, awaiting a request from the Leader to continue processing
    'FINISHED',          -- all reports have reached a terminal state (FINISHED, FAILED)
    'ABANDONED',         -- we have given up on the aggregation job entirely
    'DELETED'            -- this aggregation job has been deleted
);

-- An aggregation job, representing the aggregation of a number of client reports.
CREATE TABLE aggregation_jobs(
    id                         BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY, -- artificial ID, internal-only
    task_id                    BIGINT NOT NULL,                 -- ID of related task
    aggregation_job_id         BYTEA NOT NULL,                  -- 16-byte AggregationJobID as defined by the DAP specification
    aggregation_param          BYTEA NOT NULL,                  -- encoded aggregation parameter (opaque VDAF message)
    batch_id                   BYTEA NOT NULL,                  -- batch ID (leader-selected only; corresponds to identifier in BatchSelector)
    client_timestamp_interval  TSRANGE NOT NULL,                -- the minimal interval containing all of client timestamps included in this aggregation job
    state                      AGGREGATION_JOB_STATE NOT NULL,  -- current state of the aggregation job
    step                       INTEGER NOT NULL,                -- current step of the aggregation job
    last_request_hash          BYTEA,                           -- SHA-256 hash of the most recently received AggregationJobInitReq or AggregationJobContinueReq (helper only)

    lease_expiry             TIMESTAMP NOT NULL DEFAULT TIMESTAMP '-infinity',  -- when lease on this aggregation job expires; -infinity implies no current lease
    lease_token              BYTEA,                                             -- a value identifying the current leaseholder; NULL implies no current lease
    lease_attempts           BIGINT NOT NULL DEFAULT 0,                         -- the number of lease acquiries since the last successful lease release

    -- creation/update records
    created_at TIMESTAMP NOT NULL,  -- when the row was created
    updated_at TIMESTAMP NOT NULL,  -- when the row was last changed
    updated_by TEXT NOT NULL,       -- the name of the transaction that last updated the row

    CONSTRAINT aggregation_jobs_unique_id UNIQUE(task_id, aggregation_job_id),
    CONSTRAINT fk_task_id FOREIGN KEY(task_id) REFERENCES tasks(id) ON DELETE CASCADE
);
CREATE INDEX aggregation_jobs_state_and_lease_expiry ON aggregation_jobs(state, lease_expiry) WHERE state = 'ACTIVE';
CREATE INDEX aggregation_jobs_task_and_batch_id ON aggregation_jobs(task_id, batch_id);
CREATE INDEX aggregation_jobs_task_and_client_timestamp_interval ON aggregation_jobs USING gist (task_id, client_timestamp_interval);

-- Specifies the possible state of aggregating a single report.
CREATE TYPE REPORT_AGGREGATION_STATE AS ENUM(
    'INIT',                 -- the aggregator is ready for the aggregation initialization step
    'INIT_PROCESSING',      -- the aggregator is processing an aggregation initialization request asynchronously
    'CONTINUE',             -- the aggregator is ready for an aggregation continuation step
    'CONTINUE_PROCESSING',  -- the aggregator is processing an aggregation continuation request asynchronously
    'POLL',                 -- the aggregator is polling for completion of a previous operation
    'FINISHED',             -- the aggregator has completed the preparation process successfully
    'FAILED'                -- the aggregator has completed the preparation process unsuccessfully
);

-- An aggregation attempt for a single client report. An aggregation job logically contains a number
-- of report aggregations. A single client report might be aggregated in multiple aggregation jobs &
-- therefore have multiple associated report aggregations.
CREATE TABLE report_aggregations(
    id                  BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY, -- artificial ID, internal-only
    task_id             BIGINT NOT NULL,                    -- ID of related task
    aggregation_job_id  BIGINT NOT NULL,                    -- the aggregation job ID this report aggregation is associated with
    ord                 BIGINT NOT NULL,                    -- a value used to specify the ordering of client reports in the aggregation job
    client_report_id    BYTEA NOT NULL,                     -- the client report ID this report aggregation is associated with
    client_timestamp    TIMESTAMP NOT NULL,                 -- the client timestamp this report aggregation is associated with
    last_prep_resp      BYTEA,                              -- the last PrepareResp message sent to the Leader, to assist in replay (opaque DAP message, populated for Helper only)
    state               REPORT_AGGREGATION_STATE NOT NULL,  -- the current state of this report aggregation

    -- Additional data for state LeaderInit.
    public_extensions             BYTEA,  -- encoded sequence of public Extension messages (opaque DAP messages)
    public_share                  BYTEA,  -- the public share for the report (opaque VDAF message)
    leader_private_extensions     BYTEA,  -- encoded sequence of leader's private Extension messages (opaque DAP messages)
    leader_input_share            BYTEA,  -- encoded leader input share (opaque VDAF message)
    helper_encrypted_input_share  BYTEA,  -- encoded HPKE ciphertext of helper input share (opaque DAP message)

    -- Additional data for state LeaderContinue.
    leader_prep_transition  BYTEA,  -- the current VDAF prepare transition (opaque VDAF message)

    -- Additional data for state LeaderPoll.
    leader_prep_state    BYTEA,  -- the current prepare state (opaque VDAF message)
    leader_output_share  BYTEA,  -- the leader's recovered output share (opaque VDAF message)

    -- Additional data for state HelperInitProcessing.
    prepare_init  BYTEA,                  -- the preparation initialization message received from the Leader (opaque DAP message)
    require_taskbind_extension  BOOLEAN,  -- is the taskprov extension required?

    -- Additional data for state HelperContinue & HelperContinueProcessing.
    helper_prep_state  BYTEA,  -- the current VDAF prepare state (opaque VDAF message)

    -- Additional data for state HelperContinueProcessing.
    prepare_continue  BYTEA,  -- the preparation continuation message received from the Leader (opaque VDAF message)

    -- Additional data for state Failed.
    error_code  SMALLINT,  -- error code corresponding to a DAP ReportShareError value

    -- creation/update records
    created_at TIMESTAMP NOT NULL,  -- when the row was created
    updated_at TIMESTAMP NOT NULL,  -- when the row was last changed
    updated_by TEXT NOT NULL,       -- the name of the transaction that last updated the row

    CONSTRAINT report_aggregations_unique_ord UNIQUE(task_id, aggregation_job_id, ord),
    CONSTRAINT fk_task_id FOREIGN KEY (task_id) REFERENCES tasks (id) ON DELETE CASCADE,
    CONSTRAINT fk_aggregation_job_id FOREIGN KEY(aggregation_job_id) REFERENCES aggregation_jobs(id) ON DELETE CASCADE
);
CREATE INDEX report_aggregations_aggregation_job_id_index ON report_aggregations(aggregation_job_id);
CREATE INDEX report_aggregations_client_report_id_index ON report_aggregations(client_report_id);

-- Specifies the possible states of a batch aggregation.
CREATE TYPE BATCH_AGGREGATION_STATE AS ENUM(
    'AGGREGATING',  -- this batch aggregation has not been collected & permits further aggregation
    'COLLECTED',    -- this batch aggregation has been collected & no longer permits aggregation
    'SCRUBBED'      -- this batch aggregation has been scrubbed of data and can no longer be used
);

-- Information on aggregation for a single batch. This information may be incremental if the VDAF
-- supports incremental aggregation. Each batch's aggregation is sharded via the `ord` column.
CREATE TABLE batch_aggregations(
    id                          BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,  -- artificial ID, internal-only
    task_id                     BIGINT NOT NULL,                   -- the task ID
    batch_identifier            BYTEA NOT NULL,                    -- encoded batch mode-specific batch identifier (corresponds to identifier in BatchSelector)
    batch_interval              TSRANGE,                           -- batch interval, as a TSRANGE, populated only for time-interval tasks. (will always match batch_identifier)
    aggregation_param           BYTEA NOT NULL,                    -- the aggregation parameter (opaque VDAF message)
    ord                         BIGINT NOT NULL,                   -- the index of this batch aggregation shard, over (task ID, batch_identifier, aggregation_param).

    client_timestamp_interval   TSRANGE NOT NULL,                  -- the minimal interval containing all of client timestamps included in this batch aggregation
    state                       BATCH_AGGREGATION_STATE NOT NULL,  -- the current state of this batch aggregation
    aggregate_share             BYTEA,                             -- the (possibly-incremental) aggregate share; populated unless report_count is 0 or the batch aggregation has been scrubbed.
    report_count                BIGINT,                            -- the (possibly-incremental) client report count; populated unless the batch aggregation has been scrubbed.
    checksum                    BYTEA,                             -- the (possibly-incremental) checksum; populated unless the batch aggregation has been scrubbed.
    aggregation_jobs_created    BIGINT,                            -- the number of aggregation jobs that have been created for this batch; populated unless the batch aggregation has been scrubbed.
    aggregation_jobs_terminated BIGINT,                            -- the number of aggregation jobs that have been terminated (finished or abandoned) for this batch; populated unless the batch aggregation has been scrubbed.

    -- creation/update records
    created_at TIMESTAMP NOT NULL,  -- when the row was created
    updated_at TIMESTAMP NOT NULL,  -- when the row was last changed
    updated_by TEXT NOT NULL,       -- the name of the transaction that last updated the row

    CONSTRAINT batch_aggregations_unique_task_id_batch_id_aggregation_param UNIQUE(task_id, batch_identifier, aggregation_param, ord),
    CONSTRAINT fk_task_id FOREIGN KEY(task_id) REFERENCES tasks(id) ON DELETE CASCADE
);
CREATE INDEX batch_aggregations_gc_time ON batch_aggregations(task_id, UPPER(COALESCE(batch_interval, client_timestamp_interval)));

-- Specifies the possible state of a collection job.
CREATE TYPE COLLECTION_JOB_STATE AS ENUM(
    'START',        -- this collection job is waiting for reports to be aggregated
    'FINISHED',     -- this collection job has run successfully and is ready to be retrieved by the collector
    'ABANDONED',    -- this collection job has been abandoned & will never be run again
    'DELETED'       -- this collection job has been deleted
);

-- The leader's view of collect requests from the Collector.
CREATE TABLE collection_jobs(
    id                         BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,  -- artificial ID, internal-only
    task_id                    BIGINT NOT NULL,                -- the task ID being collected
    collection_job_id          BYTEA NOT NULL,                 -- 16 byte identifier used by collector to refer to this job
    query                      BYTEA NOT NULL,                 -- encoded batch mode-specific query (corresponds to Query)
    aggregation_param          BYTEA NOT NULL,                 -- the aggregation parameter (opaque VDAF message)
    batch_identifier           BYTEA NOT NULL,                 -- encoded batch mode-specific batch identifier (corresponds to identifier in BatchSelector)
    batch_interval             TSRANGE,                        -- batch interval, as a TSRANGE, populated only for time-interval tasks. (will always match batch_identifier)
    state                      COLLECTION_JOB_STATE NOT NULL,  -- the current state of this collection job
    report_count               BIGINT,                         -- the number of reports included in this collection job (only if in state FINISHED)
    client_timestamp_interval  TSRANGE,                        -- the minimal interval containing the reports included in this collection job, aligned to the task's time precision (only if in state FINISHED)
    helper_aggregate_share     BYTEA,                          -- the helper's encrypted aggregate share (HpkeCiphertext, only if in state FINISHED)
    leader_aggregate_share     BYTEA,                          -- the leader's unencrypted aggregate share (opaque VDAF message, only if in state FINISHED)

    step_attempts  BIGINT NOT NULL DEFAULT 0,  -- the number of attempts to step the collection job without making progress, regardless of whether the lease was successfully released or not

    lease_expiry    TIMESTAMP NOT NULL DEFAULT TIMESTAMP '-infinity',  -- when lease on this collection job expires; -infinity implies no current lease
    lease_token     BYTEA,                                             -- a value identifying the current leaseholder; NULL implies no current lease
    lease_attempts  BIGINT NOT NULL DEFAULT 0,                         -- the number of lease acquiries since the last successful lease release

    -- creation/update records
    created_at  TIMESTAMP NOT NULL,  -- when the row was created
    updated_at  TIMESTAMP NOT NULL,  -- when the row was last changed
    updated_by  TEXT NOT NULL,       -- the name of the transaction that last updated the row

    CONSTRAINT collection_jobs_unique_id UNIQUE(task_id, collection_job_id),
    CONSTRAINT fk_task_id FOREIGN KEY(task_id) REFERENCES tasks(id) ON DELETE CASCADE
);
CREATE INDEX collection_jobs_task_id_batch_id ON collection_jobs(task_id, batch_identifier);
-- TODO(#224): verify that this index is optimal for purposes of acquiring collection jobs.
CREATE INDEX collection_jobs_state_and_lease_expiry ON collection_jobs(state, lease_expiry) WHERE state = 'START';
CREATE INDEX collection_jobs_interval_containment_index ON collection_jobs USING gist (task_id, batch_interval);

-- The helper's view of aggregate share jobs.
CREATE TABLE aggregate_share_jobs(
    id                      BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY, -- artificial ID, internal-only
    task_id                 BIGINT NOT NULL,    -- the task ID being collected
    batch_identifier        BYTEA NOT NULL,     -- encoded batch mode-specific batch identifier (corresponds to identifier in BatchSelector)
    batch_interval          TSRANGE,            -- batch interval, as a TSRANGE, populated only for time-interval tasks. (will always match batch_identifier)
    aggregation_param       BYTEA NOT NULL,     -- the aggregation parameter (opaque VDAF message)
    helper_aggregate_share  BYTEA NOT NULL,     -- the helper's unencrypted aggregate share
    report_count            BIGINT NOT NULL,    -- the count of reports included helper_aggregate_share
    checksum                BYTEA NOT NULL,     -- the checksum over the reports included in helper_aggregate_share

    -- creation/update records
    created_at TIMESTAMP NOT NULL,  -- when the row was created
    updated_by TEXT NOT NULL,       -- the name of the transaction that last updated the row

    CONSTRAINT aggregate_share_jobs_unique_task_id_batch_id_aggregation_param UNIQUE(task_id, batch_identifier, aggregation_param),
    CONSTRAINT fk_task_id FOREIGN KEY(task_id) REFERENCES tasks(id) ON DELETE CASCADE
);
CREATE INDEX aggregate_share_jobs_interval_containment_index ON aggregate_share_jobs USING gist (task_id, batch_interval);

-- Specifies the possible state of an outstanding batch.
CREATE TYPE OUTSTANDING_BATCH_STATE AS ENUM(
    'FILLING',  -- this outstanding batch is still being considered for additional reports
    'FILLED'    -- this outstanding batch has received enough reports, no more are necessary
);

-- The leader's view of outstanding batches, which are batches which have not yet started
-- collection. Used for leader-selected tasks only.
CREATE TABLE outstanding_batches(
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY, -- artificial ID, internal-only
    task_id BIGINT NOT NULL, -- the task ID containing the batch
    batch_id BYTEA NOT NULL, -- 32-byte BatchID as defined by the DAP specification.
    time_bucket_start TIMESTAMP,

    state OUTSTANDING_BATCH_STATE NOT NULL DEFAULT 'FILLING',  -- the current state of this outstanding batch

    -- creation/update records
    created_at TIMESTAMP NOT NULL,  -- when the row was created
    updated_by TEXT NOT NULL,       -- the name of the transaction that last updated the row

    CONSTRAINT outstanding_batches_unique_task_id_batch_id UNIQUE(task_id, batch_id),
    CONSTRAINT fk_task_id FOREIGN KEY(task_id) REFERENCES tasks(id) ON DELETE CASCADE
);
CREATE INDEX outstanding_batches_task_id_and_time_bucket_start ON outstanding_batches(task_id, time_bucket_start) WHERE state = 'FILLING';
