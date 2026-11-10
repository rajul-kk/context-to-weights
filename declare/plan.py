from dataclasses import dataclass

SCHEMES = ("token", "channel_keys")


@dataclass(frozen=True)
class AttentionPlan:
    kind: str
    kv_bits: int
    kv_scheme: str

    @property
    def recorded_scheme(self):
        return self.kv_scheme if self.kv_bits else "none"


def plan_attention(modes, kv_bits=0, kv_scheme="token"):
    if kv_scheme not in SCHEMES:
        raise SystemExit(f"unknown kv scheme {kv_scheme!r}, expected one of {SCHEMES}")
    if kv_bits < 0:
        raise SystemExit(f"--kv-bits must be 0 or positive, got {kv_bits}")
    probing = [m for m in modes if m.startswith("attention")]
    if probing and kv_bits:
        raise SystemExit(
            f"--kv-bits {kv_bits} cannot be combined with the attention modes {probing}: the "
            f"probe and the quantizer are separate attention implementations and a run uses "
            f"exactly one. Run the attention modes without --kv-bits.")
    if probing:
        return AttentionPlan("probe", 0, kv_scheme)
    if kv_bits:
        return AttentionPlan("quant", kv_bits, kv_scheme)
    return AttentionPlan("repeat", 0, kv_scheme)
