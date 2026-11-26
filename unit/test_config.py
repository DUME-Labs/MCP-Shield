import pytest

from mcp_shield.config import ConfigError, ExplainerProvider, SecurityConfig, config_from_dict, load_config
from mcp_shield.paths import config_path

TOML = """
taint_policy = "strict"
fpr_target = 0.02

[taint]
allow_flows = ["filesystem -> github", "* -> notes"]

[explainer]
provider = "llama_cpp"
url = "http://127.0.0.1:8090/v1"

[tools.shell-tools.run_shell]
mode = "shell"

[tools.filesystem.read_file]
roots = ["~/workspace"]
privilege = "read"
field_types = { target = "path" }

[[listeners]]
name = "postgres"
listen = "127.0.0.1:8801"
upstream = "http://localhost:3000/mcp"

[dashboard]
port = 9000
origin_allowlist = ["http://localhost:5173"]
"""


def test_defaults_are_the_canonical_fields():
    c = SecurityConfig()
    assert (c.classifier_model, c.taint_policy, c.fpr_target, c.explainer) == (
        "mcp-shield-clf",
        "standard",
        0.005,
        ExplainerProvider.NONE,
    )
    assert c.fail_open is False


def test_missing_file_means_defaults():
    assert load_config().taint_policy == "standard"


def test_full_file(shield_home):
    config_path().parent.mkdir(parents=True)
    config_path().write_text(TOML, encoding="utf-8")
    c = load_config()
    assert c.taint_policy == "strict" and c.fpr_target == 0.02
    assert c.explainer is ExplainerProvider.LLAMA_CPP and c.explainer_url.endswith("/v1")
    assert c.flow_allowed("filesystem", "github") and c.flow_allowed("anything", "notes")
    assert not c.flow_allowed("github", "filesystem")
    assert c.tool_policy("shell-tools", "run_shell").mode == "shell"
    p = c.tool_policy("filesystem", "read_file")
    assert p.roots == ["~/workspace"] and p.privilege == "read" and p.field_types == {"target": "path"}
    assert c.tool_policy("other", "x").mode == "default"
    assert c.listeners[0].port == 8801 and c.dashboard_port == 9000
    assert c.origin_allowlist == ["http://localhost:5173"]
    assert c.warnings == []


@pytest.mark.parametrize(
    "raw, msg",
    [
        ({"taint_policy": "lenient"}, "taint_policy"),
        ({"fpr_target": 0.9}, "fpr_target"),
        ({"taint": {"allow_flows": ["no arrow"]}}, "allow_flows"),
        ({"tools": {"s": {"t": {"privilege": "admin"}}}}, "privilege"),
        ({"tools": {"s": {"t": {"mode": "yolo"}}}}, "mode"),
        ({"tools": {"s": {"t": {"field_types": {"x": "blob"}}}}}, "field_types"),
        ({"listeners": [{"name": "x", "upstream": "http://a", "listen": "0.0.0.0:80"}]}, "127.0.0.1"),
        ({"listeners": [{"name": "x"}]}, "listeners"),
    ],
)
def test_bad_values_fail_closed(raw, msg):
    with pytest.raises(ConfigError, match=msg):
        config_from_dict(raw)


def test_unknown_keys_warn():
    c = config_from_dict({"surprise": 1, "tools": {"s": {"t": {"colour": "red"}}}})
    assert any("surprise" in w for w in c.warnings) and any("colour" in w for w in c.warnings)


def test_wildcard_server_tool_policy():
    c = config_from_dict({"tools": {"*": {"send_email": {"requires_approval": True}}}})
    assert c.tool_policy("mail", "send_email").requires_approval


def test_bad_toml(shield_home):
    config_path().parent.mkdir(parents=True)
    config_path().write_text("taint_policy = ", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config()
