PROJECT_NAMES = [
    "orchid-gateway", "lantern-api", "basalt-worker", "meridian-sync",
    "quartz-ingest", "cobalt-router", "juniper-store", "tessera-auth",
]

SERVICES = [
    "billing", "notifications", "search-index", "media-transcode",
    "user-profile", "audit-log", "feature-flags", "rate-limiter",
]

FACT_TEMPLATES = [
    {
        "key": "db_engine",
        "statement": "We settled on {value} as the primary datastore for {service}.",
        "question": "Which datastore did we settle on for {service}?",
        "values": ["PostgreSQL 16", "CockroachDB 24.1", "MySQL 8.4", "SQLite 3.45", "ScyllaDB 6.0"],
    },
    {
        "key": "auth_header",
        "statement": "The auth token travels in the {value} header, not Authorization.",
        "question": "Which header carries the auth token?",
        "values": ["X-Session-Key", "X-Meridian-Token", "X-Tessera-Auth", "X-Client-Assert"],
    },
    {
        "key": "rate_limit",
        "statement": "The public rate limit for {service} is fixed at {value}.",
        "question": "What is the public rate limit for {service}?",
        "values": ["500 req/min", "1200 req/min", "80 req/sec", "45 req/sec", "9000 req/hour"],
    },
    {
        "key": "retry_policy",
        "statement": "Retries use {value}; anything else was rejected in review.",
        "question": "What retry policy was agreed on?",
        "values": [
            "exponential backoff capped at 30s",
            "three fixed retries at 2s intervals",
            "jittered backoff with a 12s ceiling",
            "no retries at all",
        ],
    },
    {
        "key": "deploy_target",
        "statement": "{project} deploys to {value} and nowhere else.",
        "question": "Where does {project} deploy to?",
        "values": ["eu-west-2", "us-east-1", "ap-southeast-3", "sa-east-1"],
    },
    {
        "key": "queue_name",
        "statement": "Events for {service} land on the queue named {value}.",
        "question": "Which queue receives {service} events?",
        "values": ["evt.sable.v3", "evt.harrow.v1", "evt.pallas.v2", "evt.dover.v4"],
    },
    {
        "key": "config_flag",
        "statement": "The kill switch is the flag {value}; flipping it disables writes.",
        "question": "What is the name of the kill-switch flag?",
        "values": ["ENABLE_WRITE_PATH", "GUARD_MUTATIONS", "ALLOW_PERSIST", "WRITE_FUSE_OPEN"],
    },
    {
        "key": "owner",
        "statement": "{value} owns {service} and signs off on every schema change.",
        "question": "Who owns {service}?",
        "values": ["the Platform team", "the Ingest team", "the Reliability guild", "the Data pod"],
    },
    {
        "key": "timeout",
        "statement": "The upstream timeout for {service} is {value}, agreed after the incident.",
        "question": "What is the upstream timeout for {service}?",
        "values": ["2500 ms", "800 ms", "15 s", "4 s"],
    },
    {
        "key": "version_pin",
        "statement": "{project} pins its client library to version {value} until the migration lands.",
        "question": "Which client library version is {project} pinned to?",
        "values": ["3.11.2", "0.42.0", "7.0.1", "12.4.9"],
    },
]

FILLER_USER = [
    "Can you walk me through how the {service} module handles a cold start?",
    "The linter is complaining about unused imports in the {service} package again.",
    "I rebased onto main and the {service} tests are flaky now.",
    "What's a reasonable batch size for the {service} backfill?",
    "Should we split the {service} handler into two files?",
    "The CI runner for {project} is timing out on the integration suite.",
    "Does {service} need its own connection pool or can it share?",
    "I want to add structured logging to {service}, any objections?",
    "How do we feature-gate the new {service} endpoint during rollout?",
    "The docs for {service} are stale, want me to rewrite them?",
    "Is there a reason {service} still uses the old serializer?",
    "Any thoughts on moving {service} behind the shared gateway?",
]

FILLER_ASSISTANT = [
    "Cold start in {service} warms the pool lazily, so the first request pays the connection cost. Preloading at boot would fix it.",
    "Those imports came in with the refactor. Safe to drop them, nothing reflects over that module.",
    "Flakiness there is usually the shared fixture. Give each test its own temp directory and it settles.",
    "Start around 500 rows per batch for {service}, then tune once you see the lock contention.",
    "Splitting {service} is reasonable once the file passes about 400 lines. Right now it is still readable.",
    "The integration suite in {project} spends most of its time on container setup. Cache the image layer.",
    "{service} can share the pool for now. Give it a dedicated one only if you see queueing.",
    "Structured logging is fine. Keep the field names consistent with the rest of the codebase.",
    "Gate it with the existing flag system and roll out to internal traffic first.",
    "Rewriting the {service} docs is worth it, they predate the last two interface changes.",
    "The old serializer stayed because of a downstream consumer. Check whether it is still alive.",
    "Moving {service} behind the gateway is the direction we want, just not mid-quarter.",
]

CLOSERS = [
    "Sounds good, I'll pick that up.",
    "Makes sense, thanks.",
    "Alright, noted.",
    "Fine by me.",
]
