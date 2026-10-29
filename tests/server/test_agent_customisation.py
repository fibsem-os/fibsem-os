from fibsem.ai.agent.catalog import CATALOG, GROUP_TITLES
from fibsem.server.agent_customisation import CATEGORIES, description


def test_every_tool_group_is_in_one_category():
    groups = [group for groups in CATEGORIES.values() for group in groups]
    assert len(groups) == len(set(groups))
    assert set(groups) == set(GROUP_TITLES)
    assert {tool.name.split("_")[0] for tool in CATALOG} <= set(groups)


def test_a_description_is_read_as_a_line(tmp_path):
    path = tmp_path / "SKILL.md"
    path.write_text(
        "---\nname: x\ndescription: Read it first: then the rest\n---\n# X\n"
    )
    assert description(path) == "Read it first: then the rest"
    assert description(tmp_path / "missing.md") == ""
