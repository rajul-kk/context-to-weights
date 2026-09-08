SKILL_SPECS = [
    {
        "name": "veltrix-cache",
        "category": "storage",
        "summary": "Write and read entries in the Veltrix regional cache.",
        "fn_write": "vx_stash",
        "fn_read": "vx_fetch",
        "args": ["bucket", "key", "payload", "ttl_s"],
        "rule": "ttl_s is mandatory and must not exceed 86400; Veltrix rejects a write without it.",
        "error": "VX_TTL_MISSING",
        "objects": ["session tokens", "rendered thumbnails", "pricing tables", "feature payloads"],
    },
    {
        "name": "harrowdb",
        "category": "storage",
        "summary": "Query the HarrowDB append-only ledger.",
        "fn_write": "harrow_append",
        "fn_read": "harrow_scan",
        "args": ["ledger", "record", "actor", "checkpoint"],
        "rule": "every append must carry an actor; scans without a checkpoint are refused.",
        "error": "HARROW_NO_ACTOR",
        "objects": ["audit records", "settlement rows", "policy changes", "access grants"],
    },
    {
        "name": "pallas-http",
        "category": "network",
        "summary": "Call upstream services through the Pallas egress client.",
        "fn_write": "pallas_send",
        "fn_read": "pallas_probe",
        "args": ["route", "body", "deadline_ms", "idem_key"],
        "rule": "deadline_ms defaults to nothing and must be set explicitly; retries require idem_key.",
        "error": "PALLAS_NO_DEADLINE",
        "objects": ["webhook deliveries", "payment captures", "profile syncs", "index refreshes"],
    },
    {
        "name": "sable-queue",
        "category": "network",
        "summary": "Publish and drain messages on the Sable event bus.",
        "fn_write": "sable_emit",
        "fn_read": "sable_drain",
        "args": ["topic", "envelope", "partition_hint", "dedupe_window_s"],
        "rule": "partition_hint must be a stable string per entity, never a random value.",
        "error": "SABLE_UNSTABLE_HINT",
        "objects": ["order events", "user signups", "device heartbeats", "billing ticks"],
    },
    {
        "name": "quarrybuild",
        "category": "tooling",
        "summary": "Drive the Quarry incremental build system.",
        "fn_write": "quarry_target",
        "fn_read": "quarry_explain",
        "args": ["label", "inputs", "toolchain", "sandbox"],
        "rule": "sandbox must be strict for anything published; loose sandboxes are local-only.",
        "error": "QUARRY_LOOSE_PUBLISH",
        "objects": ["release binaries", "docs bundles", "container layers", "test fixtures"],
    },
    {
        "name": "tessera-auth",
        "category": "tooling",
        "summary": "Mint and verify Tessera capability tokens.",
        "fn_write": "tessera_mint",
        "fn_read": "tessera_verify",
        "args": ["subject", "scopes", "audience", "not_after"],
        "rule": "scopes must be a sorted list; Tessera hashes them and an unsorted list fails verification.",
        "error": "TESSERA_SCOPE_ORDER",
        "objects": ["service identities", "CI runners", "partner clients", "admin sessions"],
    },
    {
        "name": "obsidian-flags",
        "category": "tooling",
        "summary": "Read and flip Obsidian feature flags.",
        "fn_write": "obs_flip",
        "fn_read": "obs_read",
        "args": ["flag", "state", "cohort", "reason"],
        "rule": "every flip needs a reason string; Obsidian writes it to the audit trail.",
        "error": "OBS_NO_REASON",
        "objects": ["rollout gates", "kill switches", "experiment arms", "regional toggles"],
    },
    {
        "name": "cinder-metrics",
        "category": "network",
        "summary": "Emit and query Cinder time-series metrics.",
        "fn_write": "cinder_point",
        "fn_read": "cinder_range",
        "args": ["series", "value", "labels", "resolution_s"],
        "rule": "labels must be low-cardinality; Cinder drops a series once it exceeds 200 label values.",
        "error": "CINDER_CARDINALITY",
        "objects": ["latency samples", "queue depth", "cache hit rate", "error counts"],
    },
]

CATEGORIES = sorted({s["category"] for s in SKILL_SPECS})

STEMS = ["vel", "har", "pal", "sab", "quar", "tess", "obs", "cin", "dro", "mar",
         "kel", "zan", "fen", "gild", "corv", "brae", "nyx", "olm", "perr", "strand",
         "lum", "vask", "orr", "thal", "wren", "ryke", "solm", "cadd", "ilex", "porth"]
TAILS = ["trix", "row", "las", "ble", "ry", "era", "idian", "der", "vale", "kos"]

DOMAINS = {
    "storage": ["store", "ledger", "vault", "index", "depot"],
    "network": ["mesh", "relay", "bus", "gateway", "stream"],
    "tooling": ["forge", "runner", "registry", "planner", "lint"],
}

VERBS = {
    "storage": [("stash", "fetch"), ("append", "scan"), ("put", "get"),
                ("commit", "lookup"), ("seal", "open")],
    "network": [("send", "probe"), ("emit", "drain"), ("push", "poll"),
                ("dispatch", "trace"), ("publish", "consume")],
    "tooling": [("mint", "verify"), ("flip", "inspect"), ("target", "explain"),
                ("register", "resolve"), ("apply", "audit")],
}

HANDLES = {
    "storage": ["bucket", "ledger", "vault", "shard", "namespace"],
    "network": ["route", "topic", "channel", "endpoint", "lane"],
    "tooling": ["label", "subject", "flag", "target", "profile"],
}

FIELDS = {
    "storage": ["key", "payload", "record", "blob", "version", "digest"],
    "network": ["body", "envelope", "frame", "headers", "cursor"],
    "tooling": ["inputs", "scopes", "state", "cohort", "toolchain"],
}

GUARDS = [
    ("{arg} is mandatory and must not exceed {limit}; {Brand} rejects a write without it.",
     "{ABBR}_{UARG}_MISSING", ["ttl_s", "budget_s", "lease_s", "window_s"]),
    ("every {Brand} write must carry {arg}; reads without it are refused.",
     "{ABBR}_NO_{UARG}", ["actor", "origin", "tenant", "owner"]),
    ("{arg} defaults to nothing on {Brand} and must be set explicitly; retries require it.",
     "{ABBR}_NO_{UARG}", ["deadline_ms", "timeout_ms", "idem_key", "trace_id"]),
    ("{arg} must be a sorted list; {Brand} hashes it and an unsorted list fails.",
     "{ABBR}_{UARG}_ORDER", ["scopes", "grants", "claims", "tags"]),
    ("{arg} must be low-cardinality; {Brand} drops the series past 200 values.",
     "{ABBR}_{UARG}_CARDINALITY", ["labels", "dimensions", "facets"]),
    ("{arg} must be a stable string per entity on {Brand}, never a random value.",
     "{ABBR}_UNSTABLE_{UARG}", ["partition_hint", "shard_key", "affinity"]),
    ("{Brand} requires {arg} to be strict for anything published; loose values stay local-only.",
     "{ABBR}_LOOSE_{UARG}", ["sandbox", "isolation", "mode"]),
    ("{arg} must be a monotonic counter; {Brand} refuses a value it has already seen.",
     "{ABBR}_{UARG}_REPLAY", ["sequence", "revision", "epoch"]),
]

OBJECTS = {
    "storage": ["session tokens", "rendered thumbnails", "pricing tables", "audit records",
                "settlement rows", "feature payloads", "access grants", "policy changes"],
    "network": ["webhook deliveries", "payment captures", "order events", "device heartbeats",
                "profile syncs", "billing ticks", "index refreshes", "user signups"],
    "tooling": ["release binaries", "docs bundles", "container layers", "rollout gates",
                "kill switches", "CI runners", "experiment arms", "test fixtures"],
}


def _abbrev(stem, used):
    for n in (2, 3, 4, 5):
        cand = stem[:n].upper()
        if cand not in used:
            used.add(cand)
            return cand
    n = 2
    while f"{stem[:2].upper()}{n}" in used:
        n += 1
    used.add(f"{stem[:2].upper()}{n}")
    return f"{stem[:2].upper()}{n}"


def generate_specs(n, seed=0, include_handwritten=True):
    import random

    rng = random.Random(seed)
    specs = list(SKILL_SPECS) if include_handwritten else []
    names = {s["name"] for s in specs}
    fns = {s["fn_write"] for s in specs} | {s["fn_read"] for s in specs}
    errors = {s["error"] for s in specs}
    abbrevs = {s["fn_write"].split("_")[0].upper() for s in specs}

    brands = [a + b for a in STEMS for b in TAILS]
    rng.shuffle(brands)
    cats = list(DOMAINS)
    i = 0
    while len(specs) < n and i < len(brands):
        brand = brands[i]
        i += 1
        cat = cats[len(specs) % len(cats)]
        name = f"{brand}-{rng.choice(DOMAINS[cat])}"
        if name in names:
            continue
        abbr = _abbrev(brand, abbrevs)
        w, r = rng.choice(VERBS[cat])
        fn_w, fn_r = f"{brand[:5]}_{w}", f"{brand[:5]}_{r}"
        if fn_w in fns or fn_r in fns:
            continue
        tmpl, err_tmpl, arg_pool = GUARDS[len(specs) % len(GUARDS)]
        guard = rng.choice(arg_pool)
        handle = rng.choice(HANDLES[cat])
        fields = rng.sample(FIELDS[cat], 2)
        error = err_tmpl.format(ABBR=abbr, UARG=guard.upper().replace("_MS", "").rstrip("_S"))
        if error in errors:
            continue
        specs.append({
            "name": name,
            "category": cat,
            "summary": f"Operate the {brand.capitalize()} {name.split('-')[-1]}.",
            "fn_write": fn_w,
            "fn_read": fn_r,
            "args": [handle] + fields + [guard],
            "rule": tmpl.format(arg=guard, Brand=brand.capitalize(),
                                limit=rng.choice(["86400", "3600", "512", "10000"])),
            "error": error,
            "objects": rng.sample(OBJECTS[cat], 4),
        })
        names.add(name)
        fns.update({fn_w, fn_r})
        errors.add(error)
    if len(specs) < n:
        raise ValueError(f"could only build {len(specs)} distinct skills, asked for {n}")
    return specs[:n]
