import pytest

from mcp_shield.cascade.argpolicy import (
    FIELD_COMMAND,
    FIELD_PATH,
    FIELD_TEXT,
    FIELD_URL,
    RULES_BOTH,
    RULES_POSIX,
    RULES_WINDOWS,
    check_path,
    check_url,
    field_type_for,
    metachar_hits,
    scan_arguments,
)
from mcp_shield.cascade.privilege import annotation_class, name_class, privilege_of
from mcp_shield.config import ToolPolicy

P = ToolPolicy()


class TestFieldTypes:
    @pytest.mark.parametrize(
        "key, expected",
        [
            ("path", FIELD_PATH),
            ("file_path", FIELD_PATH),
            ("filePath", FIELD_PATH),
            ("output_dir", FIELD_PATH),
            ("command", FIELD_COMMAND),
            ("cmd", FIELD_COMMAND),
            ("shell_command", FIELD_COMMAND),
            ("url", FIELD_URL),
            ("webhook_url", FIELD_URL),
            ("baseUri", FIELD_URL),
            ("query", FIELD_TEXT),
            ("body", FIELD_TEXT),
            ("content", FIELD_TEXT),
        ],
    )
    def test_by_name(self, key, expected):
        assert field_type_for(key, None, P, key) == expected

    def test_schema_format_marks_url(self):
        assert field_type_for("target", {"type": "string", "format": "uri"}, P, "target") == FIELD_URL

    def test_config_wins(self):
        policy = ToolPolicy(field_types={"path": "text"})
        assert field_type_for("path", None, policy, "path") == FIELD_TEXT


class TestMetachars:
    def test_posix_set(self):
        assert metachar_hits("ls; rm -rf /", RULES_POSIX) == [";"]
        assert "$(" in metachar_hits("echo $(id)", RULES_POSIX)
        assert "&&" in metachar_hits("a && b", RULES_POSIX)
        assert metachar_hits("report_Q&A.md", RULES_POSIX) == []  # a lone & is not a POSIX separator here

    def test_windows_set(self):
        assert "&" in metachar_hits("dir & calc", RULES_WINDOWS)
        assert "^" in metachar_hits("c^alc", RULES_WINDOWS)
        assert "%VAR%" in metachar_hits("type %USERPROFILE%\\x", RULES_WINDOWS)

    def test_http_mode_uses_both(self):
        hits = metachar_hits("a & b ${X}", RULES_BOTH)
        assert "&" in hits and "${" in hits


class TestPaths:
    def test_traversal(self):
        assert ("traversal", "'..' segment") in check_path("../../etc/passwd", RULES_POSIX, [], local=False)
        assert check_path("..\\..\\x", RULES_WINDOWS, [], local=False)
        assert check_path("%2e%2e/secret", RULES_POSIX, [], local=False)
        assert check_path("docs/notes..md", RULES_POSIX, [], local=False) == []

    def test_posix_backslash_is_not_a_separator(self):
        assert check_path("..\\x", RULES_POSIX, [], local=False) == []

    def test_roots(self):
        roots = ["/work/project"]
        assert check_path("/work/project/src/a.py", RULES_POSIX, roots, local=False) == []
        assert check_path("/work/projectX/a.py", RULES_POSIX, roots, local=False)
        assert check_path("/etc/passwd", RULES_POSIX, roots, local=False)
        assert check_path("src/a.py", RULES_POSIX, roots, local=False) == []

    def test_windows_roots_case_insensitive_and_prefixes(self):
        roots = ["C:\\Work"]
        assert check_path("c:\\work\\a.txt", RULES_WINDOWS, roots, local=False) == []
        assert check_path("\\\\?\\C:\\Work\\a.txt", RULES_WINDOWS, roots, local=False) == []
        assert check_path("C:/Work/../Windows/win.ini", RULES_WINDOWS, roots, local=False)
        assert check_path("D:\\Work\\a.txt", RULES_WINDOWS, roots, local=False)

    def test_windows_specials(self):
        kinds = lambda v: {k for k, _ in check_path(v, RULES_WINDOWS, [], local=False)}  # noqa: E731
        assert "stream" in kinds("C:\\x\\file.txt:secret")
        assert "stream" in kinds("C:\\x\\file.txt::$DATA")
        assert "device_path" in kinds("\\\\.\\PhysicalDrive0")
        assert "outside_root" in kinds("\\\\attacker\\share\\x")
        assert kinds("C:\\x\\file.txt") == set()


class TestUrls:
    def test_ok(self):
        assert check_url("https://api.example.com/v1?q=1", P) == []

    @pytest.mark.parametrize(
        "url, needle",
        [
            ("file:///etc/passwd", "scheme"),
            ("https://trusted.com@evil.com/", "'@'"),
            ("https://example.com/a b", "whitespace"),
            ("https://example.com\\@evil.com", "backslash"),
            ("http://169.254.169.254/latest/meta-data/", "metadata"),
            ("http://[::ffff:169.254.169.254]/", "metadata"),
            ("http:///nohost", "host"),
        ],
    )
    def test_smuggling(self, url, needle):
        assert any(needle in p for p in check_url(url, P))

    def test_host_allowlist(self):
        policy = ToolPolicy(url_hosts=["*.example.com", "api.github.com"])
        assert check_url("https://docs.example.com/x", policy) == []
        assert check_url("https://api.github.com/x", policy) == []
        assert check_url("https://evil.com/x", policy)
        assert check_url("https://example.com.evil.com/x", policy)


class TestScan:
    schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"},
            "command": {"type": "string"},
            "links": {"type": "array", "items": {"type": "string", "format": "uri"}},
        },
    }

    def test_typed_field_blocks(self):
        f = scan_arguments({"path": "../../.ssh/id_rsa"}, self.schema, P, RULES_POSIX, local=False)
        assert f and all(x.block for x in f)

    def test_text_field_annotates(self):
        f = scan_arguments({"content": "run `make` | tee log"}, self.schema, P, RULES_POSIX, local=False)
        assert f and not any(x.block for x in f)

    def test_array_items_inherit_type(self):
        f = scan_arguments(
            {"links": ["https://ok.example", "file:///etc/shadow"]}, self.schema, P, RULES_POSIX, local=False
        )
        assert [x.path for x in f if x.block] == ["links[1]"]

    def test_shell_mode_disables_operator_block(self):
        shell = ToolPolicy(mode="shell")
        assert scan_arguments({"command": "ls | wc -l"}, self.schema, shell, RULES_POSIX, local=False) == []
        assert scan_arguments({"command": "ls | wc -l"}, self.schema, P, RULES_POSIX, local=False)[0].block

    def test_nested_objects(self):
        f = scan_arguments({"opts": {"file_path": "/etc/../etc/passwd"}}, None, P, RULES_POSIX, local=False)
        assert f[0].path == "opts.file_path" and f[0].block


class TestPrivilege:
    @pytest.mark.parametrize(
        "name, cls",
        [
            ("run_shell", "execute"),
            ("execute_query", "execute"),
            ("bash", "execute"),
            ("shellExec", "execute"),
            ("send_email", "egress"),
            ("fetch_url", "egress"),
            ("post_message", "egress"),
            ("write_file", "write"),
            ("delete_issue", "write"),
            ("updateRecord", "write"),
            ("get_issue", "read"),
            ("list_files", "read"),
            ("search_code", "read"),
            ("read_file", "read"),
            ("frobnicate", None),
        ],
    )
    def test_name_rules(self, name, cls):
        assert name_class(name) == cls

    def test_unmatched_defaults_to_write(self):
        assert privilege_of({"name": "frobnicate"}, "frobnicate", P) == "write"

    def test_annotations_only_raise(self):
        tool = {"annotations": {"readOnlyHint": True, "openWorldHint": True}}
        assert privilege_of(tool, "get_page", P) == "egress"
        lying = {"annotations": {"readOnlyHint": True, "destructiveHint": False}}
        assert privilege_of(lying, "execute_shell", P) == "execute"

    def test_destructive_hint(self):
        assert annotation_class({"destructiveHint": True}) == "write"
        assert annotation_class({}) is None

    def test_config_raises_but_never_lowers(self):
        assert privilege_of({}, "get_page", ToolPolicy(privilege="egress")) == "egress"
        assert privilege_of({}, "run_query", ToolPolicy(privilege="read")) == "execute"
        assert privilege_of({}, "frobnicate", ToolPolicy(privilege="read")) == "read"
